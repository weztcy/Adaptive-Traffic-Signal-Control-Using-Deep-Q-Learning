"""
Tests for evaluation module.
"""

from src.evaluation.evaluate import run_episode
from src.environment.traffic_intersection import TrafficIntersectionEnv
from src.baselines.fixed_time_controller import FixedTimeController


def test_run_episode_metrics():
    config = {
        "simulation": {
            "time_step": 1.0,
            "decision_step": 5.0,
            "episode_duration": 60.0,
            "min_green_time": 10.0,
            "max_green_time": 60.0,
            "yellow_time": 4.0,
        }
    }
    env = TrafficIntersectionEnv(config=config, scenario="low", seed=42)
    ctrl = FixedTimeController()

    res = run_episode(env, "fixed_time", ctrl, seed=42)

    assert "avg_waiting_time" in res
    assert "avg_queue_length" in res
    assert "total_throughput" in res
    assert "total_delay" in res
    assert "max_queue_length" in res
    assert res["total_throughput"] >= 0
    assert res["avg_waiting_time"] >= 0.0
