"""
Tests for Baseline Controllers.
"""

from src.environment.traffic_intersection import TrafficIntersectionEnv
from src.baselines.fixed_time_controller import FixedTimeController, ActuatedController


def test_fixed_time_controller_run():
    env = TrafficIntersectionEnv(scenario="medium", seed=42)
    env.reset()

    controller = FixedTimeController(green_ns=20.0, green_ew=20.0, yellow_time=4.0)

    phase_changes = 0
    last_phase = env.current_phase

    for _ in range(25):
        action = controller.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if env.current_phase != last_phase:
            phase_changes += 1
            last_phase = env.current_phase

    assert phase_changes > 0, "FixedTimeController should switch phases cyclically."
    assert env.total_departed > 0, "Vehicles should depart during green phases."


def test_actuated_controller_run():
    env = TrafficIntersectionEnv(scenario="medium", seed=42)
    env.reset()

    controller = ActuatedController(min_green=10.0, max_green=60.0, queue_threshold_ratio=1.2)

    for _ in range(20):
        action = controller.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        assert action in (0, 1)

    assert env.total_arrived > 0
