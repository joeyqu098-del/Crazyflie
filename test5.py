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
ARENA_LENGTH_M = 1.00
ARENA_WIDTH_M = 0.50

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

# Control loop
CONTROL_PERIOD_S = 0.10

# Unexpected landing detection
UNEXPECTED_LANDING_HEIGHT_M = 0.12
UNEXPECTED_LANDING_HOLD_S = 0.8

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
            self.on_box = True
            self.entry = (sensors.x, sensors.y)
            self._last_edge_t = now
            print(f"  >>> BOX ENTRY at ({sensors.x:.2f}, {sensors.y:.2f}) "
                  f"(zrange step {step:+.2f} m)")

        elif self.on_box and step > BOX_EDGE_STEP_M:
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
            self.on_box = False
            self.on_box_points.append(self.entry)
            self.on_box_points.append((sensors.x, sensors.y))
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


# ============================================================
# ARENA EXPLORATION
# ============================================================

def sweep_row(mc, sensors, box_tracker, landing_watch, direction):
    """
    Sweep sideways across the current row.
    direction: +1 = left, -1 = right.
    Ends at the wall (side sensor), after ARENA_WIDTH_M of travel,
    or on timeout.
    """
    start_y = sensors.y
    start_t = time.time()
    max_time = 1.5 * ARENA_WIDTH_M / SIDEWAYS_SPEED_MPS
    min_zrange = sensors.zrange

    while time.time() - start_t < max_time:
        box_tracker.update(sensors)
        landing_watch.check(sensors)
        min_zrange = min(min_zrange, sensors.zrange)

        wall = sensors.left if direction > 0 else sensors.right
        if wall < WALL_STOP_DISTANCE_M:
            break
        if abs(sensors.y - start_y) >= ARENA_WIDTH_M:
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

    # Tuning aid: lowest downward range seen this row
    print(f"  zrange min this row: {min_zrange:.3f} m "
          f"(last: {sensors.zrange:.3f} m)")


def advance_row(mc, sensors, box_tracker, landing_watch):
    """
    Step forward one row spacing.
    Returns False if the far wall is reached.
    """
    start_x = sensors.x
    start_t = time.time()
    max_time = 3.0 * LANE_SPACING_M / FORWARD_SPEED_MPS

    while sensors.x - start_x < LANE_SPACING_M:
        if time.time() - start_t > max_time:
            break

        box_tracker.update(sensors)
        landing_watch.check(sensors)

        if sensors.front < CRITICAL_DISTANCE_M:
            mc.start_linear_motion(0.0, 0.0, 0.0)
            return False

        mc.start_linear_motion(
            FORWARD_SPEED_MPS,
            0.0,
            compute_vertical_correction(sensors),
        )
        time.sleep(CONTROL_PERIOD_S)

    mc.start_linear_motion(0.0, 0.0, 0.0)
    time.sleep(0.2)
    return True


def explore_arena(mc, sensors, box_tracker, landing_watch):
    """
    Lawnmower sweep across the WIDTH, advancing forward one row at a time:

        row 1:  ---------->
                          |  forward
        row 2:  <---------
                |  forward
        row 3:  ---------->
    """
    print("\nStarting row-by-row sweep...")

    number_of_rows = int(ARENA_LENGTH_M / LANE_SPACING_M) + 1

    # First sweep goes toward the side with more room
    direction = 1 if sensors.left >= sensors.right else -1

    for row in range(number_of_rows):
        side = "LEFT" if direction > 0 else "RIGHT"
        print(f"\nRow {row + 1}/{number_of_rows}: sweeping {side}")

        sweep_row(mc, sensors, box_tracker, landing_watch, direction)
        box_tracker.finish_row(sensors)

        if row == number_of_rows - 1:
            break

        if not advance_row(mc, sensors, box_tracker, landing_watch):
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
    """
    cx, cy = center
    dx = cx - sensors.x
    dy = cy - sensors.y

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

        log_conf.start()

        # Give logging time to initialize
        time.sleep(0.5)

        box_tracker = BoxTracker()
        landing_watch = LandingWatch()

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

                explore_arena(mc, sensors, box_tracker, landing_watch)

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

                    fly_to_box_center(mc, sensors, box_tracker, center)

                    print("\nHovering over estimated box center...")

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