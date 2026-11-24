"""
Training script for Deep Q-Network (DQN) on Adaptive Traffic Signal Control.
Includes smoke-test mode, validation checkpointing, and metrics tracking.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List
import yaml
import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.baselines.fixed_time_controller import FixedTimeController
from src.environment.traffic_intersection import TrafficIntersectionEnv


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_agent(
    agent: DQNAgent,
    config: Dict[str, Any],
    eval_seeds: List[int],
    scenario: str = "medium",
    duration: float = 1800.0,
) -> Dict[str, float]:
    """Run validation across multiple seeds with deterministic policy."""
    val_waits = []
    val_queues = []
    val_throughputs = []
    val_rewards = []

    val_cfg = dict(config)
    val_cfg["simulation"] = dict(config.get("simulation", {}))
    val_cfg["simulation"]["episode_duration"] = duration

    for seed in eval_seeds:
        env = TrafficIntersectionEnv(config=val_cfg, scenario=scenario, seed=seed)
        obs, info = env.reset(seed=seed)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        val_waits.append(info["avg_departed_wait"])
        val_queues.append(np.mean(env.history["ns_queue"]) + np.mean(env.history["ew_queue"]))
        val_throughputs.append(info["total_departed"])
        val_rewards.append(ep_reward)

    return {
        "val_avg_wait": float(np.mean(val_waits)),
        "val_avg_queue": float(np.mean(val_queues)),
        "val_throughput": float(np.mean(val_throughputs)),
        "val_reward": float(np.mean(val_rewards)),
    }


def train(
    config_path: str = "config/default.yaml",
    smoke_test: bool = False,
    episodes_override: int = None,
    save_dir: str = "models",
    metrics_dir: str = "results/metrics",
):
    config = load_config(config_path)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    dqn_cfg = config.get("dqn", {})
    sim_cfg = config.get("simulation", {})

    seed = dqn_cfg.get("seed", 42)
    np.random.seed(seed)

    if smoke_test:
        episodes = 5
        episode_duration = 600.0  # 10 minutes sim
        val_duration = 300.0
        val_freq = 2
        val_seeds = [101, 102]
        epsilon_decay_steps = 500
        print("[SMOKE TEST] Running quick 5-episode test...")
    else:
        episodes = episodes_override or dqn_cfg.get("train_episodes", 100)
        episode_duration = sim_cfg.get("episode_duration", 3600.0)
        val_duration = 1800.0
        val_freq = dqn_cfg.get("eval_frequency", 10)
        val_seeds = dqn_cfg.get("eval_seeds", [100, 200, 300])
        epsilon_decay_steps = dqn_cfg.get("epsilon_decay_steps", 10000)

    train_cfg = dict(config)
    train_cfg["simulation"] = dict(sim_cfg)
    train_cfg["simulation"]["episode_duration"] = episode_duration

    agent = DQNAgent(
        state_dim=dqn_cfg.get("state_dim", 12),
        action_dim=dqn_cfg.get("action_dim", 2),
        hidden_dims=dqn_cfg.get("hidden_dims", [128, 128]),
        learning_rate=float(dqn_cfg.get("learning_rate", 0.0005)),
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        epsilon_start=float(dqn_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(dqn_cfg.get("epsilon_end", 0.05)),
        epsilon_decay_steps=epsilon_decay_steps,
        replay_buffer_capacity=int(dqn_cfg.get("replay_buffer_capacity", 50000)),
        batch_size=int(dqn_cfg.get("batch_size", 64)),
        target_update_interval=int(dqn_cfg.get("target_update_interval", 400)),
        tau=float(dqn_cfg.get("tau", 0.005)),
        use_soft_update=bool(dqn_cfg.get("use_soft_update", False)),
        seed=seed,
    )

    env = TrafficIntersectionEnv(config=train_cfg, scenario="medium", seed=seed)

    history = {
        "episode": [],
        "reward": [],
        "avg_loss": [],
        "epsilon": [],
        "avg_wait_time": [],
        "avg_queue_length": [],
        "throughput": [],
        "max_queue": [],
        "phase_switches": [],
        "val_episodes": [],
        "val_avg_wait": [],
        "val_avg_queue": [],
        "val_reward": [],
    }

    best_val_wait = float("inf")
    best_model_path = os.path.join(save_dir, "best_dqn_model.pt")
    start_time = time.time()

    print(f"=== Starting DQN Training: {episodes} episodes ===")
    for ep in range(1, episodes + 1):
        # Vary scenario during training to foster generalization
        scenario_choice = "medium" if ep % 3 != 0 else ("high" if ep % 2 == 0 else "low")
        env.scenario = scenario_choice

        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        losses = []
        done = False

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = agent.step(obs, action, reward, next_obs, done)
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            ep_reward += reward

        mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        avg_wait = info["avg_departed_wait"]
        avg_queue = float(np.mean(env.history["ns_queue"]) + np.mean(env.history["ew_queue"])) if env.history["ns_queue"] else 0.0

        history["episode"].append(ep)
        history["reward"].append(float(ep_reward))
        history["avg_loss"].append(mean_loss)
        history["epsilon"].append(float(agent.epsilon))
        history["avg_wait_time"].append(float(avg_wait))
        history["avg_queue_length"].append(float(avg_queue))
        history["throughput"].append(int(info["total_departed"]))
        history["max_queue"].append(int(info["max_queue_length"]))
        history["phase_switches"].append(int(info["phase_switches"]))

        print(
            f"Ep {ep:3d}/{episodes} [{scenario_choice.upper():6s}] | "
            f"Rew: {ep_reward:7.1f} | Loss: {mean_loss:6.4f} | Eps: {agent.epsilon:.3f} | "
            f"Wait: {avg_wait:5.1f}s | AvgQ: {avg_queue:4.1f} | Thr: {info['total_departed']:4d}"
        )

        # Validation check
        if ep % val_freq == 0 or ep == episodes:
            val_metrics = validate_agent(
                agent=agent,
                config=config,
                eval_seeds=val_seeds,
                scenario="medium",
                duration=val_duration,
            )
            history["val_episodes"].append(ep)
            history["val_avg_wait"].append(val_metrics["val_avg_wait"])
            history["val_avg_queue"].append(val_metrics["val_avg_queue"])
            history["val_reward"].append(val_metrics["val_reward"])

            print(
                f"  >>> VALIDATION @ Ep {ep}: ValWait={val_metrics['val_avg_wait']:.2f}s | "
                f"ValQ={val_metrics['val_avg_queue']:.2f} | ValRew={val_metrics['val_reward']:.1f}"
            )

            # Checkpoint best model based on validation average waiting time (lower is better)
            if val_metrics["val_avg_wait"] < best_val_wait:
                best_val_wait = val_metrics["val_avg_wait"]
                metadata = {
                    "best_episode": ep,
                    "best_val_wait": best_val_wait,
                    "val_metrics": val_metrics,
                    "config": config,
                    "seed": seed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                agent.save(best_model_path, metadata=metadata)
                print(f"  >>> [CHECKPOINT] Saved new best model to {best_model_path} (Wait: {best_val_wait:.2f}s)")

    elapsed = time.time() - start_time
    print(f"=== Training completed in {elapsed:.1f}s ===")

    # Ensure best model exists
    if not os.path.exists(best_model_path):
        agent.save(best_model_path, metadata={"note": "final_model", "best_val_wait": best_val_wait})

    # Save training history & metadata
    history_path = os.path.join(metrics_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    metadata_path = os.path.join(save_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "total_episodes": episodes,
                "best_val_wait": best_val_wait,
                "elapsed_seconds": elapsed,
                "seed": seed,
                "model_path": best_model_path,
                "history_path": history_path,
            },
            f,
            indent=2,
        )

    print(f"Saved training history to {history_path}")
    print(f"Saved training metadata to {metadata_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Traffic Signal Agent")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    args = parser.parse_args()

    train(config_path=args.config, smoke_test=args.smoke_test, episodes_override=args.episodes)
