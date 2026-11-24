"""
Comprehensive Evaluation Module.
Conducts matched-seed, fair evaluations comparing DQN against Fixed-Time and Actuated baselines
across Low, Medium, High, and Dynamic Changing traffic demand scenarios.
"""

import argparse
import json
import os
from typing import Any, Dict, List
import yaml
import numpy as np
import pandas as pd

from src.agents.dqn_agent import DQNAgent
from src.baselines.fixed_time_controller import FixedTimeController, ActuatedController
from src.environment.traffic_intersection import TrafficIntersectionEnv


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_episode(
    env: TrafficIntersectionEnv,
    controller_type: str,
    controller: Any,
    seed: int,
) -> Dict[str, Any]:
    """Execute a single evaluation episode with full state and metric tracking."""
    obs, info = env.reset(seed=seed)
    done = False
    episode_reward = 0.0

    while not done:
        if controller_type == "dqn":
            action = controller.select_action(obs, evaluate=True)
        else:
            action = controller.select_action(env)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    # Compute aggregate metrics
    ns_q = env.history["ns_queue"]
    ew_q = env.history["ew_queue"]
    total_q_series = [n + e for n, e in zip(ns_q, ew_q)]
    avg_queue_len = float(np.mean(total_q_series)) if total_q_series else 0.0

    departed_waits = env.departed_waiting_times
    avg_wait_time = float(np.mean(departed_waits)) if len(departed_waits) > 0 else 0.0
    total_delay = float(np.sum(departed_waits)) if len(departed_waits) > 0 else 0.0

    return {
        "controller": controller_type,
        "scenario": env.scenario,
        "seed": seed,
        "avg_waiting_time": avg_wait_time,
        "avg_queue_length": avg_queue_len,
        "total_throughput": int(info["total_departed"]),
        "total_delay": total_delay,
        "max_queue_length": int(info["max_queue_length"]),
        "phase_switches": int(info["phase_switches"]),
        "total_reward": float(episode_reward),
        "total_arrived": int(info["total_arrived"]),
        "history": {
            "time": env.history["time"],
            "ns_queue": env.history["ns_queue"],
            "ew_queue": env.history["ew_queue"],
            "phase": env.history["phase"],
            "departures": env.history["departures"],
            "avg_wait": env.history["avg_wait"],
        },
    }


def evaluate(
    config_path: str = "config/default.yaml",
    model_path: str = "models/best_dqn_model.pt",
    eval_seeds: List[int] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    config = load_config(config_path)
    if eval_seeds is None:
        eval_seeds = config.get("dqn", {}).get("eval_seeds", [101, 202, 303, 404, 505])

    scenarios = ["low", "medium", "high", "changing"]

    metrics_dir = os.path.join(output_dir, "metrics")
    sims_dir = os.path.join(output_dir, "simulations")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(sims_dir, exist_ok=True)

    # Initialize controllers
    dqn_agent = DQNAgent(
        state_dim=config["dqn"]["state_dim"],
        action_dim=config["dqn"]["action_dim"],
        hidden_dims=config["dqn"]["hidden_dims"],
    )
    if os.path.exists(model_path):
        dqn_agent.load(model_path)
        print(f"Loaded trained DQN model from {model_path}")
    else:
        print(f"[WARNING] Model file {model_path} not found! Using initialized weights.")

    fixed_time_ctrl = FixedTimeController(
        green_ns=config["baseline"]["fixed_time"]["green_ns"],
        green_ew=config["baseline"]["fixed_time"]["green_ew"],
        yellow_time=config["baseline"]["fixed_time"]["yellow_time"],
    )

    actuated_ctrl = ActuatedController(
        min_green=config["baseline"]["actuated"]["min_green"],
        max_green=config["baseline"]["actuated"]["max_green"],
        queue_threshold_ratio=config["baseline"]["actuated"]["queue_threshold_ratio"],
    )

    controllers = {
        "fixed_time": ("Fixed-Time", fixed_time_ctrl),
        "actuated": ("Actuated", actuated_ctrl),
        "dqn": ("DQN (Proposed)", dqn_agent),
    }

    raw_results = []
    simulation_traces = {}

    print("\n================ EVALUATION BENCHMARK ================")
    print(f"Scenarios: {scenarios}")
    print(f"Seeds: {eval_seeds} (N={len(eval_seeds)} runs per combination)")

    for scenario in scenarios:
        simulation_traces[scenario] = {}
        print(f"\n--- Scenario: {scenario.upper()} ---")

        for c_key, (c_name, c_inst) in controllers.items():
            simulation_traces[scenario][c_key] = {}
            for seed in eval_seeds:
                env = TrafficIntersectionEnv(config=config, scenario=scenario, seed=seed)
                res = run_episode(env, c_key, c_inst, seed)

                # Keep trace of first seed for detailed plotting
                if seed == eval_seeds[0]:
                    simulation_traces[scenario][c_key] = res["history"]

                # Exclude full history from flat tabular record
                res_clean = {k: v for k, v in res.items() if k != "history"}
                res_clean["controller_name"] = c_name
                raw_results.append(res_clean)

            # Summarize this controller & scenario
            c_runs = [r for r in raw_results if r["scenario"] == scenario and r["controller"] == c_key]
            mean_wait = np.mean([r["avg_waiting_time"] for r in c_runs])
            std_wait = np.std([r["avg_waiting_time"] for r in c_runs])
            mean_q = np.mean([r["avg_queue_length"] for r in c_runs])
            std_q = np.std([r["avg_queue_length"] for r in c_runs])
            mean_thr = np.mean([r["total_throughput"] for r in c_runs])
            print(
                f"  [{c_name:18s}] Wait: {mean_wait:5.2f} +/- {std_wait:4.2f}s | "
                f"Queue: {mean_q:5.2f} +/- {std_q:4.2f} | Thr: {mean_thr:6.1f} veh"
            )

    df_raw = pd.DataFrame(raw_results)

    # Save detailed per-run metrics
    raw_path = os.path.join(metrics_dir, "evaluation_detailed.json")
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2)

    # Compute aggregated statistical summary
    summary_rows = []
    grouped = df_raw.groupby(["scenario", "controller_name"])

    for (scenario, ctrl_name), group in grouped:
        summary_rows.append({
            "scenario": scenario,
            "controller": ctrl_name,
            "runs": len(group),
            "avg_waiting_time_mean": float(group["avg_waiting_time"].mean()),
            "avg_waiting_time_std": float(group["avg_waiting_time"].std()),
            "avg_queue_length_mean": float(group["avg_queue_length"].mean()),
            "avg_queue_length_std": float(group["avg_queue_length"].std()),
            "total_throughput_mean": float(group["total_throughput"].mean()),
            "total_throughput_std": float(group["total_throughput"].std()),
            "total_delay_mean": float(group["total_delay"].mean()),
            "total_delay_std": float(group["total_delay"].std()),
            "max_queue_length_mean": float(group["max_queue_length"].mean()),
            "max_queue_length_std": float(group["max_queue_length"].std()),
            "phase_switches_mean": float(group["phase_switches"].mean()),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Compute percentage improvements relative to Fixed-Time
    # For waiting time, queue, delay, max queue: lower is better -> (DQN - FT)/FT * 100
    df_pivot = df_summary.pivot(index="scenario", columns="controller")

    improvements = []
    for sc in scenarios:
        ft_wait = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "Fixed-Time")]["avg_waiting_time_mean"].values[0]
        dqn_wait = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "DQN (Proposed)")]["avg_waiting_time_mean"].values[0]
        wait_pct = ((dqn_wait - ft_wait) / ft_wait) * 100.0

        ft_q = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "Fixed-Time")]["avg_queue_length_mean"].values[0]
        dqn_q = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "DQN (Proposed)")]["avg_queue_length_mean"].values[0]
        q_pct = ((dqn_q - ft_q) / ft_q) * 100.0

        ft_thr = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "Fixed-Time")]["total_throughput_mean"].values[0]
        dqn_thr = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "DQN (Proposed)")]["total_throughput_mean"].values[0]
        thr_pct = ((dqn_thr - ft_thr) / ft_thr) * 100.0

        ft_delay = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "Fixed-Time")]["total_delay_mean"].values[0]
        dqn_delay = df_summary[(df_summary["scenario"] == sc) & (df_summary["controller"] == "DQN (Proposed)")]["total_delay_mean"].values[0]
        delay_pct = ((dqn_delay - ft_delay) / ft_delay) * 100.0

        improvements.append({
            "scenario": sc,
            "wait_time_change_pct": wait_pct,
            "queue_len_change_pct": q_pct,
            "throughput_change_pct": thr_pct,
            "total_delay_change_pct": delay_pct,
        })

    df_improvements = pd.DataFrame(improvements)

    # Save CSV outputs
    summary_csv_path = os.path.join(metrics_dir, "evaluation_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    improvements_csv_path = os.path.join(metrics_dir, "evaluation_improvements.csv")
    df_improvements.to_csv(improvements_csv_path, index=False)

    # Save traces for visualization
    traces_path = os.path.join(sims_dir, "simulation_traces.json")
    with open(traces_path, "w") as f:
        json.dump(simulation_traces, f, indent=2)

    print(f"\nSaved evaluation metrics to {summary_csv_path}")
    print(f"Saved improvements to {improvements_csv_path}")
    print(f"Saved simulation traces to {traces_path}")

    return {
        "raw_results": raw_results,
        "summary": summary_rows,
        "improvements": improvements,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Controllers")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model", type=str, default="models/best_dqn_model.pt")
    args = parser.parse_args()

    evaluate(config_path=args.config, model_path=args.model)
