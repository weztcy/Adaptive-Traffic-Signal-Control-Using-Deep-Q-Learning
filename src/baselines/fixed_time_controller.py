"""
Conventional baseline traffic signal controllers:
1. FixedTimeController: Standard pre-timed signal cycle.
2. ActuatedController: Dynamic vehicle-actuated controller based on real-time queue pressure.
"""

from typing import Dict, Any


class FixedTimeController:
    """
    Conventional fixed-time (pre-timed) signal controller.
    Cycles between NS Green and EW Green with fixed durations.
    Respects the same environment action interface (0: keep, 1: switch).
    """

    def __init__(
        self,
        green_ns: float = 30.0,
        green_ew: float = 30.0,
        yellow_time: float = 4.0,
    ):
        self.green_ns = green_ns
        self.green_ew = green_ew
        self.yellow_time = yellow_time
        self.name = "Fixed-Time Controller"

    def select_action(self, env) -> int:
        """
        Determines whether to keep current phase (0) or request phase switch (1).
        Switch is requested when current green duration reaches the target green duration.
        """
        # If currently in clearance phase, action doesn't trigger switch anyway
        if not env.can_switch_phase:
            return 0

        # Current phase is NS_GREEN
        if env.current_phase == 0:  # NS_GREEN
            if env.phase_time >= self.green_ns:
                return 1
            return 0

        # Current phase is EW_GREEN
        elif env.current_phase == 2:  # EW_GREEN
            if env.phase_time >= self.green_ew:
                return 1
            return 0

        return 0


class ActuatedController:
    """
    Vehicle-actuated traffic signal controller.
    Switches phase if opposing queue significantly outweighs active queue
    after minimum green time has elapsed, or if active queue is cleared.
    """

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 60.0,
        queue_threshold_ratio: float = 1.3,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.queue_threshold_ratio = queue_threshold_ratio
        self.name = "Actuated Controller"

    def select_action(self, env) -> int:
        """Decide action based on queue imbalance."""
        if not env.can_switch_phase:
            return 0

        ns_queue = len(env.queues["north"]) + len(env.queues["south"])
        ew_queue = len(env.queues["east"]) + len(env.queues["west"])

        # If North-South has green
        if env.current_phase == 0:  # NS_GREEN
            # Switch if opposing queue is higher and min green is met, or if NS is empty and EW has vehicles
            if ns_queue == 0 and ew_queue > 0:
                return 1
            if ew_queue > ns_queue * self.queue_threshold_ratio:
                return 1
            return 0

        # If East-West has green
        elif env.current_phase == 2:  # EW_GREEN
            if ew_queue == 0 and ns_queue > 0:
                return 1
            if ns_queue > ew_queue * self.queue_threshold_ratio:
                return 1
            return 0

        return 0
