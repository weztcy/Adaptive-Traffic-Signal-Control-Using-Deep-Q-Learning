"""
Tests for TrafficIntersectionEnv and TrafficGenerator.
"""

import numpy as np
import pytest
from src.environment.traffic_generator import TrafficGenerator, Vehicle
from src.environment.traffic_intersection import TrafficIntersectionEnv, SignalPhase


def test_vehicle_creation_and_step():
    veh = Vehicle(vehicle_id=1, approach="north", arrival_time=10.0)
    assert veh.id == 1
    assert veh.approach == "north"
    assert veh.arrival_time == 10.0
    assert veh.waiting_time == 0.0

    veh.step(1.0)
    assert veh.waiting_time == 1.0
    assert veh.total_delay == 1.0

    veh.departure_time = 15.0
    assert veh.total_delay == 5.0


def test_traffic_generator_scenarios():
    gen = TrafficGenerator(scenario="low", seed=42)
    rates = gen.get_arrival_rates(0.0)
    assert "north" in rates and rates["north"] > 0

    arrivals = gen.generate_arrivals(dt=10.0, current_time=0.0)
    assert len(arrivals) == 4
    for app in ["north", "south", "east", "west"]:
        assert isinstance(arrivals[app], list)

    # Dynamic changing scenario
    gen_dyn = TrafficGenerator(scenario="changing", seed=42)
    rates_early = gen_dyn.get_arrival_rates(100.0)
    rates_rush = gen_dyn.get_arrival_rates(1200.0)
    assert rates_rush["north"] > rates_early["north"]


def test_environment_reset_and_observation():
    env = TrafficIntersectionEnv(scenario="medium", seed=42)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (12,)
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()
    assert (obs >= 0.0).all()
    assert (obs <= 1.0).all()

    assert info["current_phase"] == int(SignalPhase.NS_GREEN)
    assert info["total_arrived"] == 0
    assert info["total_departed"] == 0
    assert info["current_total_queue"] == 0


def test_no_conflicting_green_phases():
    """Safety check: NS and EW must NEVER both be green simultaneously."""
    env = TrafficIntersectionEnv(scenario="high", seed=42)
    env.reset()

    for step_idx in range(50):
        # alternate actions
        action = 1 if step_idx % 3 == 0 else 0
        obs, reward, terminated, truncated, info = env.step(action)

        green_apps = env._get_green_approaches()
        # Verify no conflicting green approaches
        has_ns = any(a in env.NS_APPROACHES for a in green_apps)
        has_ew = any(a in env.EW_APPROACHES for a in green_apps)
        assert not (has_ns and has_ew), f"Safety violation at step {step_idx}: conflicting green phases active!"


def test_minimum_green_enforcement():
    """Verify that a phase switch does not occur until min_green_time has elapsed."""
    config = {
        "simulation": {
            "time_step": 1.0,
            "decision_step": 1.0,
            "min_green_time": 10.0,
            "max_green_time": 60.0,
            "yellow_time": 4.0,
        }
    }
    env = TrafficIntersectionEnv(config=config, scenario="low", seed=42)
    env.reset()

    # Try to switch at t=1 (before min_green=10)
    obs, reward, terminated, truncated, info = env.step(action=1)
    assert env.current_phase == SignalPhase.NS_GREEN, "Phase switched before min_green_time!"

    # Advance until min_green is met
    for _ in range(9):
        env.step(action=0)

    assert env.phase_time >= 10.0
    assert env.can_switch_phase is True

    # Now request switch
    env.step(action=1)
    assert env.current_phase == SignalPhase.NS_CLEARANCE, "Should transition to clearance phase!"


def test_queue_non_negativity_and_departures():
    env = TrafficIntersectionEnv(scenario="medium", seed=42)
    env.reset()

    for _ in range(30):
        env.step(action=0)
        for app in env.APPROACHES:
            assert len(env.queues[app]) >= 0

    assert env.total_arrived >= env.total_departed


def test_reward_sanity_check():
    """Verify that a state with fewer queues and less delay receives a higher reward."""
    env = TrafficIntersectionEnv(scenario="medium", seed=42)
    env.reset()

    # Empty intersection reward
    empty_reward = env._compute_reward()

    # Add vehicles to queue
    for app in env.APPROACHES:
        for i in range(10):
            env.queues[app].append(Vehicle(vehicle_id=i, approach=app, arrival_time=0.0))
            env.queues[app][-1].waiting_time = 30.0

    congested_reward = env._compute_reward()
    assert empty_reward > congested_reward, (
        f"Reward sanity check failed: empty {empty_reward} should be > congested {congested_reward}"
    )
