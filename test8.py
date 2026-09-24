"""
Crazyflie Stage 2 - Row-by-Row Arena Sweep

Mission:
    1. Connect to Crazyflie
    2. Take off
    3. Sweep the arena row by row (sideways across the width,
       then one step forward, then sideways back)
    4. Detect the box from steps in the raw down-facing range
       (range.zrange): a drop = box entry, a jump = box exit
    5. Estimate the center of the box
    6. Fly to the estimated center (backward moves allowed)
    7. Land

Notes:
    - The sweep only ever advances forward between rows.
    - A small backward creep is used only if something is
      critically close in front.
    - The drone launches from somewhere INSIDE the row's width,
      not at a side wall. Row 1 is split into two legs (to the
      near wall, then to the far wall) measured from the launch
      point, so the sweep doesn't overshoot. Every row after that
      is a normal full-width sweep.

Requires:
    pip install cflib

Hardware assumed:
    - Multiranger deck (front/back/left/right sensors)
    - Flow deck with z-ranger
"""

import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


# ============================================================
# CONFIGURATION
# ============================================================

URI = uri_helper.uri_from_env(
    default="radio://0/88/2M/E7E7E7E7F0"
)

# Normal flight height
FLIGHT_HEIGHT_M = 0.50

# Arena dimensions
ARENA_LENGTH_M = 3.00
ARENA_WIDTH_M = 1.00  # full width, wall to wall

# The Crazyflie launches from partway across the row's width, not
# from a side wall. These are the distances from the launch point
# to each side wall - they should add up to ARENA_WIDTH_M. Measure
# your actual setup and adjust these two if they aren't 0.25/0.75.
LAUNCH_DIST_TO_NEAR_WALL_M = 0.25  # whichever wall is closer at launch
LAUNCH_DIST_TO_FAR_WALL_M = 0.75   # the opposite wall

# Speeds
FORWARD_SPEED_MPS = 0.12
SIDEWAYS_SPEED_MPS = 0.20

# Small backward creep used ONLY when an obstacle is critically
# close in front, to avoid crashing into it.
BACKWARD_SPEED_MPS = 0.08

# Distance between rows
LANE_SPACING_M = 0.25

# Obstacle avoidance
CRITICAL_DISTANCE_M = 0.25

# Stop a sweep when the side sensor sees a wall this close
WALL_STOP_DISTANCE_M = 0.12

# Vertical safety limits
MAX_HEIGHT_DEVIATION_M = 0.35
VERTICAL_CORRECTION_MPS = 0.15

# Box detection (raw downward range steps)
BOX_EDGE_STEP_M = 0.05       # per-sample jump in zrange that counts as an edge
BOX_EDGE_DEBOUNCE_S = 0.3    # ignore further edges right after one

# A position report this far outside the known arena is almost
# certainly a bad state estimate (e.g. the Kalman filter still
# converging right after takeoff) rather than a real drone position.
# Used to reject bogus box-edge points and bogus fly-to-center moves.
POSITION_SANITY_MARGIN_M = 0.30

# Control loop
CONTROL_PERIOD_S = 0.10

# Unexpected landing detection
UNEXPECTED_LANDING_HEIGHT_M = 0.12
UNEXPECTED_LANDING_HOLD_S = 0.8

# Attitude stability: normal flight wobble is a couple of degrees.
# A sustained tilt past this usually means a collision or loss of
# control, not just noise, and is treated as an abort condition.
MAX_TILT_DEG = 25.0
UNSTABLE_ATTITUDE_HOLD_S = 0.5

logging.basicConfig(level=logging.ERROR)


# ============================================================
# SENSOR STATE
# ============================================================

class SensorState:
    """Stores the latest Crazyflie sensor readings."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.front = 999.0
        self.back = 999.0
        self.left = 999.0
        self.right = 999.0
        self.zrange = 999.0

    def update(self, data):
        self.x = data["stateEstimate.x"]
        self.y = data["stateEstimate.y"]
        self.z = data["stateEstimate.z"]

        # Multiranger / z-ranger values are in millimeters.
        self.front = data["range.front"] / 1000.0
        self.back = data["range.back"] / 1000.0
        self.left = data["range.left"] / 1000.0
        self.right = data["range.right"] / 1000.0
        self.zrange = data["range.zrange"] / 1000.0

    def update_attitude(self, data):
        # Degrees.
        self.roll = data["stabilizer.roll"]
        self.pitch = data["stabilizer.pitch"]
        self.yaw = data["stabilizer.yaw"]


# ============================================================
# VERTICAL CONTROL
# ============================================================

def compute_vertical_correction(sensors):
    """Keep the drone within a safe band around the flight height."""

    error = sensors.z - FLIGHT_HEIGHT_M

    # Safety ceiling
    if error > MAX_HEIGHT_DEVIATION_M:
        return -VERTICAL_CORRECTION_MPS

    # If we fall too low, climb back up
    elif error < -MAX_HEIGHT_DEVIATION_M:
        return VERTICAL_CORRECTION_MPS

    return 0.0


# ============================================================
# BOX TRACKER
# ============================================================

class BoxTracker:
    """
    Detect the box from steps in the raw downward range.

    stateEstimate.z can't be used: the controller climbs over the box
    and the estimate returns to the setpoint. zrange shows the step:
        drop  -> box leading edge (entering)
        jump  -> box trailing edge (leaving)
    """

    def __init__(self):
        self.on_box = False
        self.entry = None
        self.exit = None
        self._prev_z = None
        self._last_edge_t = 0.0
        self.on_box_points = []

    def update(self, sensors):
        z = sensors.zrange
        now = time.time()

        if self._prev_z is None:
            self._prev_z = z
            return

        # Only evaluate when a NEW log sample arrived
        if z == self._prev_z:
            return

        step = z - self._prev_z
        self._prev_z = z

        if now - self._last_edge_t < BOX_EDGE_DEBOUNCE_S:
            return

        if not self.on_box and step < -BOX_EDGE_STEP_M:
            if not position_is_plausible(sensors.x, sensors.y):
                print(f"  (ignoring BOX ENTRY at ({sensors.x:.2f}, {sensors.y:.2f}) "
                      f"- outside the arena, likely a stale position estimate)")
                self._last_edge_t = now
                return
            self.on_box = True
            self.entry = (sensors.x, sensors.y)
            self._last_edge_t = now
            print(f"  >>> BOX ENTRY at ({sensors.x:.2f}, {sensors.y:.2f}) "
                  f"(zrange step {step:+.2f} m)")

        elif self.on_box and step > BOX_EDGE_STEP_M:
            if not position_is_plausible(sensors.x, sensors.y):
                print(f"  (ignoring BOX EXIT at ({sensors.x:.2f}, {sensors.y:.2f}) "
                      f"- outside the arena, likely a stale position estimate; "
                      f"still waiting for a real exit)")
                self._last_edge_t = now
                return
            self.on_box = False
            self.exit = (sensors.x, sensors.y)
            self._last_edge_t = now
            self.on_box_points.append(self.entry)
            self.on_box_points.append(self.exit)
            print(f"  >>> BOX EXIT at ({sensors.x:.2f}, {sensors.y:.2f}) "
                  f"(zrange step {step:+.2f} m)")

    def finish_row(self, sensors):
        """Close out a crossing if a row ended while still over the box."""
        if self.on_box:
            exit_point = (sensors.x, sensors.y)
            if not position_is_plausible(*exit_point) or not position_is_plausible(*self.entry):
                print("  >>> Row ended while on box, but the entry or current "
                      "position looks implausible - discarding this crossing")
                self.on_box = False
                return
            self.on_box = False
            self.on_box_points.append(self.entry)
            self.on_box_points.append(exit_point)
            print("  >>> Row ended while on box - recorded partial crossing")

    def estimate_center(self):
        if not self.on_box_points:
            return None

        cx = sum(p[0] for p in self.on_box_points) / len(self.on_box_points)
        cy = sum(p[1] for p in self.on_box_points) / len(self.on_box_points)

        return cx, cy


# ============================================================
# SAFETY CHECKS
# ============================================================

def position_is_plausible(x, y):
    """
    Reject positions far outside the known arena. Used to filter out
    bad state-estimate glitches (most commonly the Kalman filter still
    converging in the first moments after takeoff) so they don't get
    treated as real box edges or a real fly-to-center target.
    """
    x_min = -POSITION_SANITY_MARGIN_M
    x_max = ARENA_LENGTH_M + POSITION_SANITY_MARGIN_M
    y_bound = ARENA_WIDTH_M + POSITION_SANITY_MARGIN_M
    return x_min <= x <= x_max and -y_bound <= y <= y_bound


def front_obstacle_critical(sensors):
    return sensors.front < CRITICAL_DISTANCE_M


class UnexpectedLanding(Exception):
    pass


class LandingWatch:
    """Raises UnexpectedLanding if the height stays very low."""

    def __init__(self):
        self._since = None

    def check(self, sensors):
        if sensors.z < UNEXPECTED_LANDING_HEIGHT_M:
            if self._since is None:
                self._since = time.time()
            elif time.time() - self._since > UNEXPECTED_LANDING_HOLD_S:
                raise UnexpectedLanding(
                    f"Estimated height is only {sensors.z:.2f} m."
                )
        else:
            self._since = None


class UnstableAttitude(Exception):
    pass


class AttitudeWatch:
    """
    Raises UnstableAttitude if roll or pitch stays past MAX_TILT_DEG.

    Brief spikes (a gust, a sharp turn) are normal and self-correct
    almost immediately, so this only trips on a *sustained* tilt,
    which is a much stronger sign of a collision or a drone that has
    lost control than of ordinary flight wobble.
    """

    def __init__(self):
        self._since = None

    def check(self, sensors):
        if abs(sensors.roll) > MAX_TILT_DEG or abs(sensors.pitch) > MAX_TILT_DEG:
            if self._since is None:
                self._since = time.time()
            elif time.time() - self._since > UNSTABLE_ATTITUDE_HOLD_S:
                raise UnstableAttitude(
                    f"Attitude unstable: roll={sensors.roll:.1f} deg, "
                    f"pitch={sensors.pitch:.1f} deg."
                )
        else:
            self._since = None


# ============================================================
# ARENA EXPLORATION
# ============================================================

def sweep_row(mc, sensors, box_tracker, landing_watch, attitude_watch, direction,
              max_distance=ARENA_WIDTH_M, reference_y=None,
              timeout_distance=None):
    """
    Sweep sideways across the current row.
    direction: +1 = left, -1 = right.
    Ends at the wall (side sensor), once travel from `reference_y`
    reaches `max_distance`, or on timeout.

    reference_y defaults to the current position, which is what every
    normal full-width row wants. Pass the row's original launch
    position instead to keep measuring distance from there across a
    direction reversal - this is how the first row's two legs (to the
    near wall, then to the far wall) both stay anchored to the actual
    launch point instead of restarting the count at the near wall.

    timeout_distance controls only the safety timeout clock; pass it
    separately from max_distance when a leg's real travel distance is
    longer than its stopping-distance cap (again, the row-1 far leg).

    Returns True if a full box crossing (entry + exit) completed
    during this leg. When that happens the leg stops immediately
    instead of continuing to the wall/timeout, and the caller should
    stop the whole sweep rather than moving on to the next row.
    """
    start_y = sensors.y if reference_y is None else reference_y
    if timeout_distance is None:
        timeout_distance = max_distance

    start_t = time.time()
    max_time = 1.5 * timeout_distance / SIDEWAYS_SPEED_MPS
    min_zrange = sensors.zrange
    box_points_before = len(box_tracker.on_box_points)
    box_found = False

    while time.time() - start_t < max_time:
        box_tracker.update(sensors)
        if len(box_tracker.on_box_points) > box_points_before:
            box_found = True
            break

        landing_watch.check(sensors)
        attitude_watch.check(sensors)
        min_zrange = min(min_zrange, sensors.zrange)

        wall = sensors.left if direction > 0 else sensors.right
        if wall < WALL_STOP_DISTANCE_M:
            break
        if abs(sensors.y - start_y) >= max_distance:
            break

        # No forward motion during the sweep (small back-off only if
        # something is critically close in front).
        vx = -BACKWARD_SPEED_MPS if front_obstacle_critical(sensors) else 0.0

        mc.start_linear_motion(
            vx,
            SIDEWAYS_SPEED_MPS * direction,
            compute_vertical_correction(sensors),
        )
        time.sleep(CONTROL_PERIOD_S)

    mc.start_linear_motion(0.0, 0.0, 0.0)
    time.sleep(0.2)

    # Tuning aid: lowest downward range seen this leg
    print(f"  zrange min this leg: {min_zrange:.3f} m "
          f"(last: {sensors.zrange:.3f} m)")

    return box_found


def advance_row(mc, sensors, box_tracker, landing_watch, attitude_watch):
    """
    Step forward one row spacing.
    Returns (reached_next_row, box_found):
      - reached_next_row is False if the far wall is reached.
      - box_found is True if a full box crossing (entry + exit)
        completed while advancing, in which case the caller should
        stop the sweep entirely.
    """
    start_x = sensors.x
    start_t = time.time()
    max_time = 3.0 * LANE_SPACING_M / FORWARD_SPEED_MPS
    box_points_before = len(box_tracker.on_box_points)

    while sensors.x - start_x < LANE_SPACING_M:
        if time.time() - start_t > max_time:
            break

        box_tracker.update(sensors)
        if len(box_tracker.on_box_points) > box_points_before:
            mc.start_linear_motion(0.0, 0.0, 0.0)
            time.sleep(0.2)
            return True, True

        landing_watch.check(sensors)
        attitude_watch.check(sensors)

        if sensors.front < CRITICAL_DISTANCE_M:
            mc.start_linear_motion(0.0, 0.0, 0.0)
            return False, False

        mc.start_linear_motion(
            FORWARD_SPEED_MPS,
            0.0,
            compute_vertical_correction(sensors),
        )
        time.sleep(CONTROL_PERIOD_S)

    mc.start_linear_motion(0.0, 0.0, 0.0)
    time.sleep(0.2)
    return True, False


def explore_arena(mc, sensors, box_tracker, landing_watch, attitude_watch):
    """
    Lawnmower sweep across the WIDTH, advancing forward one row at a time.

    Row 1 is special: the drone launches partway across the width, so
    it sweeps to the near wall first (LAUNCH_DIST_TO_NEAR_WALL_M), then
    reverses across to the far wall (LAUNCH_DIST_TO_FAR_WALL_M), both
    measured from the launch point:

        row 1:  <---X========>   (near-wall leg, then far-wall leg)
                             |  forward
        row 2:  <---------
                |  forward
        row 3:  ---------->

    Every row after that is a normal full-width sweep, alternating
    direction, same as before.

    As soon as a full box crossing (entry + exit) is detected, the
    sweep stops immediately - it does not continue on to finish the
    remaining rows.
    """
    print("\nStarting row-by-row sweep...")

    number_of_rows = int(ARENA_LENGTH_M / LANE_SPACING_M) + 1

    # Go toward whichever wall the side sensors say is nearer first.
    direction = 1 if sensors.left <= sensors.right else -1
    row_start_y = sensors.y

    for row in range(number_of_rows):
        if row == 0:
            near_side = "LEFT" if direction > 0 else "RIGHT"
            far_side = "RIGHT" if direction > 0 else "LEFT"
            print(f"\nRow 1/{number_of_rows}: launch point offset - "
                  f"{near_side} {LAUNCH_DIST_TO_NEAR_WALL_M:.2f} m to near wall, "
                  f"then {far_side} to far wall")

            # Leg 1: launch point -> near wall
            box_found = sweep_row(
                mc, sensors, box_tracker, landing_watch, attitude_watch, direction,
                max_distance=LAUNCH_DIST_TO_NEAR_WALL_M,
                reference_y=row_start_y)

            if not box_found:
                # Leg 2: near wall -> far wall (still measured from the
                # original launch point, so the cap doesn't reset)
                direction = -direction
                box_found = sweep_row(
                    mc, sensors, box_tracker, landing_watch, attitude_watch, direction,
                    max_distance=LAUNCH_DIST_TO_FAR_WALL_M,
                    reference_y=row_start_y,
                    timeout_distance=ARENA_WIDTH_M)
        else:
            side = "LEFT" if direction > 0 else "RIGHT"
            print(f"\nRow {row + 1}/{number_of_rows}: sweeping {side}")
            box_found = sweep_row(mc, sensors, box_tracker, landing_watch,
                                   attitude_watch, direction)

        box_tracker.finish_row(sensors)

        if box_found:
            print("\nBox found - stopping sweep early.")
            break

        if row == number_of_rows - 1:
            break

        advanced, box_found = advance_row(mc, sensors, box_tracker,
                                           landing_watch, attitude_watch)

        if box_found:
            print("\nBox found while advancing - stopping sweep early.")
            break

        if not advanced:
            print("  Far wall reached - ending sweep.")
            break

        direction = -direction

    print("\nArena sweep complete.")
    print(f"Recorded {len(box_tracker.on_box_points)} box edge points.")


# ============================================================
# FLY TO BOX CENTER
# ============================================================

def fly_to_box_center(mc, sensors, box_tracker, center):
    """
    Move to the estimated box center with a position-based move.
    move_distance accepts negative values, so this can go backward.

    Returns False without moving if the target is implausibly far
    away (a sign the center estimate was corrupted by a bad position
    reading), so the caller can land in place instead.
    """
    cx, cy = center
    dx = cx - sensors.x
    dy = cy - sensors.y

    max_dx = ARENA_LENGTH_M + POSITION_SANITY_MARGIN_M
    max_dy = ARENA_WIDTH_M + POSITION_SANITY_MARGIN_M
    if abs(dx) > max_dx or abs(dy) > max_dy:
        print(f"\nEstimated box center ({cx:.2f}, {cy:.2f}) is implausibly far "
              f"from the current position ({sensors.x:.2f}, {sensors.y:.2f}) - "
              f"skipping the move (this points to a bad position reading, "
              f"not a real box location).")
        return False

    print(f"\nEstimated box center: ({cx:.2f}, {cy:.2f})")
    print(f"Moving by dx={dx:+.2f} m, dy={dy:+.2f} m")

    mc.move_distance(dx, dy, 0.0, velocity=0.15)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    cflib.crtp.init_drivers()

    print("Connecting to Crazyflie...")

    with SyncCrazyflie(
        URI,
        cf=Crazyflie(rw_cache="./cache")
    ) as scf:

        print("Connected!")

        sensors = SensorState()

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        log_conf = LogConfig(name="Explore", period_in_ms=100)

        log_conf.add_variable("stateEstimate.x", "float")
        log_conf.add_variable("stateEstimate.y", "float")
        log_conf.add_variable("stateEstimate.z", "float")
        log_conf.add_variable("range.front", "uint16_t")
        log_conf.add_variable("range.back", "uint16_t")
        log_conf.add_variable("range.left", "uint16_t")
        log_conf.add_variable("range.right", "uint16_t")
        log_conf.add_variable("range.zrange", "uint16_t")

        scf.cf.log.add_config(log_conf)

        log_conf.data_received_cb.add_callback(
            lambda ts, data, lc: sensors.update(data)
        )

        # Separate log block for attitude: kept apart from the block
        # above so the two don't together exceed the Crazyflie's
        # per-block payload limit.
        attitude_log_conf = LogConfig(name="Attitude", period_in_ms=100)

        attitude_log_conf.add_variable("stabilizer.roll", "float")
        attitude_log_conf.add_variable("stabilizer.pitch", "float")
        attitude_log_conf.add_variable("stabilizer.yaw", "float")

        scf.cf.log.add_config(attitude_log_conf)

        attitude_log_conf.data_received_cb.add_callback(
            lambda ts, data, lc: sensors.update_attitude(data)
        )

        log_conf.start()
        attitude_log_conf.start()

        # Give logging time to initialize
        time.sleep(0.5)

        box_tracker = BoxTracker()
        landing_watch = LandingWatch()
        attitude_watch = AttitudeWatch()

        try:

            with MotionCommander(
                scf,
                default_height=FLIGHT_HEIGHT_M
            ) as mc:

                print("\nTaking off...")
                time.sleep(1.5)
                print(f"Flight height: {FLIGHT_HEIGHT_M:.2f} m")

                # ------------------------------------------------
                # EXPLORE
                # ------------------------------------------------

                explore_arena(mc, sensors, box_tracker, landing_watch, attitude_watch)

                # ------------------------------------------------
                # ESTIMATE BOX CENTER
                # ------------------------------------------------

                center = box_tracker.estimate_center()

                if center is None:

                    print("\nNo box detected.")
                    print("Landing at current position.")

                    mc.start_linear_motion(0.0, 0.0, 0.0)
                    time.sleep(1)

                else:

                    cx, cy = center

                    print("\nBOX CENTER ESTIMATE:")
                    print(f"    x = {cx:.3f} m")
                    print(f"    y = {cy:.3f} m")

                    moved = fly_to_box_center(mc, sensors, box_tracker, center)

                    if moved:
                        print("\nHovering over estimated box center...")
                    else:
                        print("Landing at current position instead.")

                    mc.start_linear_motion(0.0, 0.0, 0.0)
                    time.sleep(1.5)

                # ------------------------------------------------
                # LAND
                # ------------------------------------------------

                print("\nLanding...")

                # MotionCommander automatically lands when
                # the context manager exits.
                time.sleep(1)

        except UnexpectedLanding as e:

            print("\n!!! MISSION ABORTED !!!")
            print(e)
            print("The drone appears to have landed unexpectedly.")

        finally:

            log_conf.stop()

    print("\nMission complete.")


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()

