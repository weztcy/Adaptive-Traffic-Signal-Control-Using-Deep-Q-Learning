"""
Vehicle generation module for traffic simulation.
Supports Poisson arrival process for static and dynamic (changing) traffic demand scenarios.
"""

from typing import Dict, Optional, Union
import numpy as np


class Vehicle:
    """Represents an individual vehicle in the intersection queue."""

    def __init__(self, vehicle_id: int, approach: str, arrival_time: float):
        self.id = vehicle_id
        self.approach = approach
        self.arrival_time = arrival_time
        self.waiting_time = 0.0
        self.departure_time: Optional[float] = None

    @property
    def total_delay(self) -> float:
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        return self.waiting_time

    def step(self, dt: float = 1.0) -> None:
        """Accumulate waiting time while queued."""
        self.waiting_time += dt

    def __repr__(self) -> str:
        return f"Vehicle(id={self.id}, approach={self.approach}, wait={self.waiting_time:.1f}s)"


class TrafficGenerator:
    """
    Stochastic traffic generator using Poisson arrival processes.
    Generates discrete vehicle arrivals for 4 approaches: north, south, east, west.
    """

    APPROACHES = ["north", "south", "east", "west"]

    def __init__(
        self,
        scenario: str = "medium",
        scenario_config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        self.scenario = scenario
        self.scenario_config = scenario_config or {}
        self.rng = np.random.default_rng(seed)
        self.current_time = 0.0
        self.vehicle_counter = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator state and optionally update random seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_time = 0.0
        self.vehicle_counter = 0

    def get_arrival_rates(self, sim_time: float) -> Dict[str, float]:
        """
        Get arrival rates (vehicles per second) for each approach at the specified simulation time.
        """
        if self.scenario == "changing":
            # Dynamic traffic pattern:
            # 0 - 900s: Moderate balanced traffic
            # 900 - 1800s: Heavy North-South rush
            # 1800 - 2700s: Heavy East-West rush
            # 2700 - 3600s: Return to light balanced flow
            if sim_time < 900.0:
                return {"north": 0.15, "south": 0.15, "east": 0.12, "west": 0.12}
            elif sim_time < 1800.0:
                return {"north": 0.38, "south": 0.38, "east": 0.12, "west": 0.12}
            elif sim_time < 2700.0:
                return {"north": 0.12, "south": 0.12, "east": 0.35, "west": 0.35}
            else:
                return {"north": 0.12, "south": 0.12, "east": 0.10, "west": 0.10}

        rates = self.scenario_config.get("rates", {})
        if rates:
            return {
                "north": float(rates.get("north", 0.2)),
                "south": float(rates.get("south", 0.2)),
                "east": float(rates.get("east", 0.15)),
                "west": float(rates.get("west", 0.15)),
            }

        # Fallback default presets
        defaults = {
            "low": {"north": 0.10, "south": 0.10, "east": 0.07, "west": 0.07},
            "medium": {"north": 0.22, "south": 0.22, "east": 0.16, "west": 0.16},
            "high": {"north": 0.38, "south": 0.38, "east": 0.30, "west": 0.30},
        }
        return defaults.get(self.scenario, defaults["medium"])

    def generate_arrivals(self, dt: float, current_time: float) -> Dict[str, list]:
        """
        Generate new vehicle arrivals for time step dt using Poisson distribution.
        """
        self.current_time = current_time
        rates = self.get_arrival_rates(current_time)
        arrivals: Dict[str, list] = {app: [] for app in self.APPROACHES}

        for approach in self.APPROACHES:
            rate = rates[approach]
            expected_num = rate * dt
            # Poisson draw: number of arrivals in interval dt
            num_arrivals = int(self.rng.poisson(expected_num))
            for _ in range(num_arrivals):
                self.vehicle_counter += 1
                veh = Vehicle(
                    vehicle_id=self.vehicle_counter,
                    approach=approach,
                    arrival_time=current_time,
                )
                arrivals[approach].append(veh)

        return arrivals
