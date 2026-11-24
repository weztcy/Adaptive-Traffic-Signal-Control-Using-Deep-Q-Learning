"""
Traffic intersection simulation environment.
Implements a 4-way intersection with conflicting phases (NS vs EW),
clearance intervals, queue dynamics, saturation flow limits, and a Gymnasium-like interface.
"""

from collections import deque
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.environment.traffic_generator import TrafficGenerator, Vehicle


class SignalPhase(IntEnum):
    """Signal phase states for the 4-way intersection."""
    NS_GREEN = 0     # North-South Green, East-West Red
    NS_CLEARANCE = 1 # North-South Yellow / Clearance, East-West Red
    EW_GREEN = 2     # East-West Green, North-South Red
    EW_CLEARANCE = 3 # East-West Yellow / Clearance, North-South Red


class TrafficIntersectionEnv:
    """
    Simulated 4-way signalized intersection environment.
    Conforms to Gymnasium API: reset(), step(action), render(), close().
    """

    APPROACHES = ["north", "south", "east", "west"]
    NS_APPROACHES = ["north", "south"]
    EW_APPROACHES = ["east", "west"]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        scenario: str = "medium",
        seed: Optional[int] = None,
    ):
        self.config = config or {}
        sim_cfg = self.config.get("simulation", {})
        self.time_step = float(sim_cfg.get("time_step", 1.0))
        self.decision_step = float(sim_cfg.get("decision_step", 5.0))
        self.episode_duration = float(sim_cfg.get("episode_duration", 3600.0))
        self.min_green_time = float(sim_cfg.get("min_green_time", 10.0))
        self.max_green_time = float(sim_cfg.get("max_green_time", 60.0))
        self.yellow_time = float(sim_cfg.get("yellow_time", 4.0))
        self.saturation_flow_rate = float(sim_cfg.get("saturation_flow_rate", 0.55)) # veh/sec
        self.max_queue_capacity = float(sim_cfg.get("max_queue_capacity", 150.0))
        self.max_wait_norm = float(sim_cfg.get("max_wait_norm", 180.0))

        # Reward weights
        rew_cfg = self.config.get("reward", {})
        self.w_queue = float(rew_cfg.get("queue_weight", 0.04))
        self.w_wait = float(rew_cfg.get("wait_weight", 0.02))
        self.w_pressure = float(rew_cfg.get("pressure_weight", 0.03))
        self.switch_penalty = float(rew_cfg.get("switch_penalty", 1.5))
        self.emergency_penalty = float(rew_cfg.get("emergency_penalty", 5.0))

        self.scenario = scenario
        scenario_dict = self.config.get("scenarios", {}).get(scenario, {})
        self.traffic_gen = TrafficGenerator(
            scenario=scenario,
            scenario_config=scenario_dict,
            seed=seed,
        )

        # Simulation state
        self.sim_time = 0.0
        self.current_phase = SignalPhase.NS_GREEN
        self.phase_time = 0.0  # Time elapsed in current phase
        self.target_phase: Optional[SignalPhase] = None
        self.queues: Dict[str, deque[Vehicle]] = {app: deque() for app in self.APPROACHES}

        # Fractional discharge accumulators (to model continuous saturation flow rate accurately)
        self.discharge_accumulators: Dict[str, float] = {app: 0.0 for app in self.APPROACHES}

        # Episode statistics
        self.total_arrived = 0
        self.total_departed = 0
        self.departed_waiting_times: List[float] = []
        self.max_queue_length = 0
        self.total_phase_switches = 0
        self.history: Dict[str, List[float]] = {
            "time": [],
            "ns_queue": [],
            "ew_queue": [],
            "phase": [],
            "departures": [],
            "avg_wait": [],
        }

        # Observation dimension: 12 features
        # 4 queue lengths, 4 average wait times, active green phase (0 for NS, 1 for EW),
        # normalized phase time, can_switch indicator, normalized total queue
        self.state_dim = 12
        self.action_dim = 2  # 0: Keep current phase, 1: Switch phase

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self.traffic_gen.reset(seed=seed)
        else:
            self.traffic_gen.reset()

        self.sim_time = 0.0
        self.current_phase = SignalPhase.NS_GREEN
        self.phase_time = 0.0
        self.target_phase = None
        self.queues = {app: deque() for app in self.APPROACHES}
        self.discharge_accumulators = {app: 0.0 for app in self.APPROACHES}

        self.total_arrived = 0
        self.total_departed = 0
        self.departed_waiting_times = []
        self.max_queue_length = 0
        self.total_phase_switches = 0
        self.history = {
            "time": [],
            "ns_queue": [],
            "ew_queue": [],
            "phase": [],
            "departures": [],
            "avg_wait": [],
        }

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    @property
    def can_switch_phase(self) -> bool:
        """Agent can only request a switch if in green phase and min green duration elapsed."""
        in_green = self.current_phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN)
        min_met = self.phase_time >= self.min_green_time
        return in_green and min_met

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the environment by one decision step (e.g., 5 seconds).
        Action:
            0: Keep current phase
            1: Switch phase (transition via clearance interval)
        """
        assert action in (0, 1), f"Invalid action {action}. Must be 0 or 1."

        # Check if phase switch was requested and permitted
        switch_triggered = False
        if action == 1 and self.can_switch_phase:
            self._initiate_phase_switch()
            switch_triggered = True

        # Advance simulation for decision_step seconds at time_step resolution
        elapsed_in_decision = 0.0
        step_rewards = []
        while elapsed_in_decision < self.decision_step:
            self._simulation_substep(self.time_step)
            elapsed_in_decision += self.time_step
            # Compute immediate substep reward
            sub_r = self._compute_reward(switch_penalty=self.switch_penalty if (switch_triggered and elapsed_in_decision == self.time_step) else 0.0)
            step_rewards.append(sub_r)

        mean_reward = float(np.mean(step_rewards))
        terminated = self.sim_time >= self.episode_duration
        truncated = False
        obs = self._get_observation()
        info = self._get_info()

        return obs, mean_reward, terminated, truncated, info

    def _initiate_phase_switch(self) -> None:
        """Initiate transition from current green phase to clearance."""
        if self.current_phase == SignalPhase.NS_GREEN:
            self.current_phase = SignalPhase.NS_CLEARANCE
            self.target_phase = SignalPhase.EW_GREEN
            self.phase_time = 0.0
            self.total_phase_switches += 1
        elif self.current_phase == SignalPhase.EW_GREEN:
            self.current_phase = SignalPhase.EW_CLEARANCE
            self.target_phase = SignalPhase.NS_GREEN
            self.phase_time = 0.0
            self.total_phase_switches += 1

    def _simulation_substep(self, dt: float) -> None:
        """Advance physical simulation by dt seconds (1.0s)."""
        self.sim_time += dt
        self.phase_time += dt

        # Handle automatic clearance phase completion
        if self.current_phase == SignalPhase.NS_CLEARANCE:
            if self.phase_time >= self.yellow_time:
                self.current_phase = SignalPhase.EW_GREEN
                self.phase_time = 0.0
                self.target_phase = None
        elif self.current_phase == SignalPhase.EW_CLEARANCE:
            if self.phase_time >= self.yellow_time:
                self.current_phase = SignalPhase.NS_GREEN
                self.phase_time = 0.0
                self.target_phase = None
        elif self.phase_time >= self.max_green_time:
            # Enforce maximum green time to avoid starving opposing approach
            self._initiate_phase_switch()

        # 1. Generate new vehicle arrivals
        arrivals = self.traffic_gen.generate_arrivals(dt, self.sim_time)
        for app, veh_list in arrivals.items():
            self.queues[app].extend(veh_list)
            self.total_arrived += len(veh_list)

        # 2. Discharge vehicles from green approaches
        substep_departures = 0
        active_approaches = self._get_green_approaches()
        for app in self.APPROACHES:
            if app in active_approaches:
                # Green indication: discharge vehicles according to saturation flow
                self.discharge_accumulators[app] += self.saturation_flow_rate * dt
                while self.discharge_accumulators[app] >= 1.0 and len(self.queues[app]) > 0:
                    veh = self.queues[app].popleft()
                    veh.departure_time = self.sim_time
                    self.departed_waiting_times.append(veh.waiting_time)
                    self.total_departed += 1
                    substep_departures += 1
                    self.discharge_accumulators[app] -= 1.0
                # Cap accumulator if queue is empty
                if len(self.queues[app]) == 0:
                    self.discharge_accumulators[app] = min(self.discharge_accumulators[app], 0.99)
            else:
                # Red or Yellow/Clearance: no discharge allowed
                self.discharge_accumulators[app] = 0.0

        # 3. Update waiting time for all queued vehicles
        for app in self.APPROACHES:
            for veh in self.queues[app]:
                veh.step(dt)

        # Update metrics
        total_q = sum(len(self.queues[app]) for app in self.APPROACHES)
        self.max_queue_length = max(self.max_queue_length, total_q)

        # Record history periodically (every 5 seconds)
        if int(self.sim_time) % 5 == 0:
            ns_q = len(self.queues["north"]) + len(self.queues["south"])
            ew_q = len(self.queues["east"]) + len(self.queues["west"])
            avg_w = self._get_current_avg_wait()
            self.history["time"].append(self.sim_time)
            self.history["ns_queue"].append(ns_q)
            self.history["ew_queue"].append(ew_q)
            self.history["phase"].append(float(self.current_phase))
            self.history["departures"].append(substep_departures)
            self.history["avg_wait"].append(avg_w)

    def _get_green_approaches(self) -> List[str]:
        """Return approaches that currently have a valid green signal."""
        if self.current_phase == SignalPhase.NS_GREEN:
            return self.NS_APPROACHES
        elif self.current_phase == SignalPhase.EW_GREEN:
            return self.EW_APPROACHES
        # During NS_CLEARANCE or EW_CLEARANCE, yellow clearance applies - no green
        return []

    def _get_current_avg_wait(self) -> float:
        """Get the current average waiting time of queued vehicles."""
        total_veh = sum(len(self.queues[app]) for app in self.APPROACHES)
        if total_veh == 0:
            return 0.0
        total_wait = sum(sum(v.waiting_time for v in self.queues[app]) for app in self.APPROACHES)
        return total_wait / total_veh

    def _get_observation(self) -> np.ndarray:
        """
        Build the 12-dimensional normalized observation vector:
        [0..3]: Normalized queue length for North, South, East, West
        [4..7]: Normalized average wait time for North, South, East, West
        [8]: Active phase (0.0: NS green/clearance, 1.0: EW green/clearance)
        [9]: Normalized elapsed phase time
        [10]: Can-switch indicator (1.0 if can switch, 0.0 otherwise)
        [11]: Normalized total queue
        """
        obs = np.zeros(self.state_dim, dtype=np.float32)

        # 0..3: Queues
        for i, app in enumerate(self.APPROACHES):
            obs[i] = min(len(self.queues[app]) / self.max_queue_capacity, 1.0)

        # 4..7: Average wait time per approach
        for i, app in enumerate(self.APPROACHES):
            q_len = len(self.queues[app])
            if q_len > 0:
                avg_w = sum(v.waiting_time for v in self.queues[app]) / q_len
                obs[4 + i] = min(avg_w / self.max_wait_norm, 1.0)
            else:
                obs[4 + i] = 0.0

        # 8: Phase indicator (0.0 for NS, 1.0 for EW)
        if self.current_phase in (SignalPhase.NS_GREEN, SignalPhase.NS_CLEARANCE):
            obs[8] = 0.0
        else:
            obs[8] = 1.0

        # 9: Normalized phase time
        obs[9] = min(self.phase_time / self.max_green_time, 1.0)

        # 10: Can-switch flag
        obs[10] = 1.0 if self.can_switch_phase else 0.0

        # 11: Normalized total queue across intersection
        total_q = sum(len(self.queues[app]) for app in self.APPROACHES)
        obs[11] = min(total_q / (self.max_queue_capacity * 2.0), 1.0)

        return obs

    def _compute_reward(self, switch_penalty: float = 0.0) -> float:
        """
        Compute reward discouraging queue length, waiting delay, and extreme imbalance.
        Formula:
            r = - (w_q * Total_Queue + w_w * Avg_Wait + w_p * |Q_NS - Q_EW|) - P_switch - P_emerg
        """
        ns_q = len(self.queues["north"]) + len(self.queues["south"])
        ew_q = len(self.queues["east"]) + len(self.queues["west"])
        total_q = ns_q + ew_q

        avg_wait = self._get_current_avg_wait()
        pressure = abs(ns_q - ew_q)

        # Emergency penalty if any queue spills over threshold (e.g. > 80 veh)
        emerg_penalty = 0.0
        for app in self.APPROACHES:
            if len(self.queues[app]) > 80:
                emerg_penalty += self.emergency_penalty

        reward = -(
            self.w_queue * total_q
            + self.w_wait * avg_wait
            + self.w_pressure * pressure
        ) - switch_penalty - emerg_penalty

        return float(reward)

    def _get_info(self) -> Dict[str, Any]:
        """Get diagnostic information and current metrics."""
        total_q = sum(len(self.queues[app]) for app in self.APPROACHES)
        avg_departed_wait = (
            float(np.mean(self.departed_waiting_times))
            if len(self.departed_waiting_times) > 0
            else 0.0
        )
        return {
            "sim_time": self.sim_time,
            "current_phase": int(self.current_phase),
            "phase_name": self.current_phase.name,
            "total_arrived": self.total_arrived,
            "total_departed": self.total_departed,
            "current_total_queue": total_q,
            "max_queue_length": self.max_queue_length,
            "avg_departed_wait": avg_departed_wait,
            "current_avg_wait": self._get_current_avg_wait(),
            "phase_switches": self.total_phase_switches,
            "queues": {app: len(self.queues[app]) for app in self.APPROACHES},
        }

    def render(self, mode: str = "text") -> str:
        """Render a concise text representation of the intersection."""
        n_q = len(self.queues["north"])
        s_q = len(self.queues["south"])
        e_q = len(self.queues["east"])
        w_q = len(self.queues["west"])
        phase_str = self.current_phase.name
        summary = (
            f"[t={self.sim_time:.0f}s | Phase={phase_str}] "
            f"Queues: N={n_q}, S={s_q}, E={e_q}, W={w_q} | "
            f"Departed={self.total_departed}, Arrived={self.total_arrived}"
        )
        return summary
