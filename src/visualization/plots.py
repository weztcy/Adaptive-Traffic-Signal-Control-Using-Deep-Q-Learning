"""
Comprehensive visualization module for Adaptive Traffic Signal Control.
Generates publication-quality charts for training, comparative evaluation,
traffic time-series dynamics, signal phase Gantt timelines, and intersection schematics.
"""

import json
import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

COLORS = {
    "Fixed-Time": "#e74c3c",      # Red
    "Actuated": "#f39c12",        # Orange
    "DQN (Proposed)": "#2ecc71",  # Green
    "ns": "#3498db",              # Blue
    "ew": "#9b59b6",              # Purple
}


def plot_training_curves(
    history_path: str = "results/metrics/training_history.json",
    output_path: str = "results/figures/training_performance.png",
) -> None:
    """Plot comprehensive 4-panel training progress curves."""
    if not os.path.exists(history_path):
        print(f"[WARNING] History file {history_path} does not exist. Skipping training plot.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    episodes = history["episode"]
    rewards = history["reward"]
    losses = history["avg_loss"]
    epsilons = history["epsilon"]
    waits = history["avg_wait_time"]
    queues = history["avg_queue_length"]

    # Compute smoothed reward (window = 10)
    window = min(10, len(rewards))
    if window > 1:
        smoothed_reward = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    else:
        smoothed_reward = rewards

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    # 1. Episode Reward
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, color="#3498db", alpha=0.35, label="Raw Episode Reward")
    ax1.plot(episodes, smoothed_reward, color="#1f77b4", linewidth=2.5, label=f"Smoothed Reward (MA-{window})")
    if "val_episodes" in history and len(history["val_episodes"]) > 0:
        ax1.scatter(
            history["val_episodes"],
            history["val_reward"],
            color="#e74c3c",
            s=50,
            zorder=5,
            label="Validation Reward",
        )
    ax1.set_title("Training & Validation Reward Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Training Loss
    ax2 = axes[0, 1]
    ax2.plot(episodes, losses, color="#e67e22", linewidth=1.8, label="Mean Huber Loss")
    ax2.set_title("DQN Optimization Loss Curve")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Huber Loss")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Epsilon Decay
    ax3 = axes[1, 0]
    ax3.plot(episodes, epsilons, color="#9b59b6", linewidth=2.0, label="Exploration Rate (Epsilon)")
    ax3.set_title("Epsilon-Greedy Exploration Schedule")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_ylim(-0.02, 1.05)
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Operational Metrics: Waiting Time & Queue Length
    ax4 = axes[1, 1]
    line1 = ax4.plot(episodes, waits, color="#e74c3c", linewidth=2.0, label="Avg Waiting Time (s)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Avg Waiting Time (seconds)", color="#e74c3c")
    ax4.tick_params(axis="y", labelcolor="#e74c3c")

    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(episodes, queues, color="#27ae60", linewidth=2.0, linestyle="--", label="Avg Queue Length (veh)")
    ax4_twin.set_ylabel("Avg Queue Length (vehicles)", color="#27ae60")
    ax4_twin.tick_params(axis="y", labelcolor="#27ae60")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right")
    ax4.set_title("Operational Traffic Metrics During Training")
    ax4.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved training plot: {output_path}")


def plot_evaluation_metrics(
    summary_csv_path: str = "results/metrics/evaluation_summary.csv",
    output_dir: str = "results/figures",
) -> None:
    """Generate comparative scenario bar charts with error bars for all key metrics."""
    if not os.path.exists(summary_csv_path):
        print(f"[WARNING] Summary CSV {summary_csv_path} does not exist. Skipping eval plots.")
        return

    df = pd.read_csv(summary_csv_path)
    scenarios = ["low", "medium", "high", "changing"]
    controllers = ["Fixed-Time", "Actuated", "DQN (Proposed)"]
    scenario_labels = ["Low Demand", "Medium Demand", "High Demand", "Changing Demand"]

    x = np.arange(len(scenarios))
    width = 0.26

    # 1. Average Waiting Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for i, ctrl in enumerate(controllers):
        subset = df[df["controller"] == ctrl].set_index("scenario").reindex(scenarios)
        means = subset["avg_waiting_time_mean"]
        stds = subset["avg_waiting_time_std"]
        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=ctrl,
            color=COLORS.get(ctrl, "#7f8c8d"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Average Vehicle Waiting Time (seconds)")
    ax.set_title("Comparison of Average Vehicle Waiting Time by Traffic Scenario (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(title="Signal Controller", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    out_wait = os.path.join(output_dir, "waiting_time_comparison.png")
    plt.tight_layout()
    plt.savefig(out_wait, dpi=200)
    plt.close()
    print(f"Saved evaluation plot: {out_wait}")

    # 2. Average Queue Length Comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for i, ctrl in enumerate(controllers):
        subset = df[df["controller"] == ctrl].set_index("scenario").reindex(scenarios)
        means = subset["avg_queue_length_mean"]
        stds = subset["avg_queue_length_std"]
        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=ctrl,
            color=COLORS.get(ctrl, "#7f8c8d"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Average Queue Length (vehicles)")
    ax.set_title("Comparison of Average Queue Length by Traffic Scenario (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(title="Signal Controller", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    out_queue = os.path.join(output_dir, "queue_length_comparison.png")
    plt.tight_layout()
    plt.savefig(out_queue, dpi=200)
    plt.close()
    print(f"Saved evaluation plot: {out_queue}")

    # 3. Throughput Comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for i, ctrl in enumerate(controllers):
        subset = df[df["controller"] == ctrl].set_index("scenario").reindex(scenarios)
        means = subset["total_throughput_mean"]
        stds = subset["total_throughput_std"]
        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=ctrl,
            color=COLORS.get(ctrl, "#7f8c8d"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{int(round(h))}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Total Intersection Throughput (vehicles / hour)")
    ax.set_title("Comparison of Total Throughput by Traffic Scenario (Higher is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(title="Signal Controller", loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    out_thr = os.path.join(output_dir, "throughput_comparison.png")
    plt.tight_layout()
    plt.savefig(out_thr, dpi=200)
    plt.close()
    print(f"Saved evaluation plot: {out_thr}")

    # 4. Total Delay and Max Queue Length Subplots
    fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Delay
    for i, ctrl in enumerate(controllers):
        subset = df[df["controller"] == ctrl].set_index("scenario").reindex(scenarios)
        means = subset["total_delay_mean"] / 1000.0  # In thousand vehicle-seconds
        stds = subset["total_delay_std"] / 1000.0
        ax_d.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=ctrl,
            color=COLORS.get(ctrl, "#7f8c8d"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
    ax_d.set_ylabel("Total Delay (x10³ vehicle-seconds)")
    ax_d.set_title("Total Accumulated Delay (Lower is Better)")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(scenario_labels, rotation=15)
    ax_d.legend(loc="upper left")
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Max Queue
    for i, ctrl in enumerate(controllers):
        subset = df[df["controller"] == ctrl].set_index("scenario").reindex(scenarios)
        means = subset["max_queue_length_mean"]
        stds = subset["max_queue_length_std"]
        ax_m.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=ctrl,
            color=COLORS.get(ctrl, "#7f8c8d"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
    ax_m.set_ylabel("Peak Max Queue Length (vehicles)")
    ax_m.set_title("Maximum Queue Spillback Risk (Lower is Better)")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(scenario_labels, rotation=15)
    ax_m.legend(loc="upper left")
    ax_m.grid(True, linestyle="--", alpha=0.5, axis="y")

    out_delay_max = os.path.join(output_dir, "delay_and_max_queue.png")
    plt.tight_layout()
    plt.savefig(out_delay_max, dpi=200)
    plt.close()
    print(f"Saved evaluation plot: {out_delay_max}")


def plot_overall_summary(
    improvements_csv: str = "results/metrics/evaluation_improvements.csv",
    output_path: str = "results/figures/overall_performance_summary.png",
) -> None:
    """Plot relative percentage improvements of DQN over Fixed-Time baseline."""
    if not os.path.exists(improvements_csv):
        print(f"[WARNING] Improvements CSV {improvements_csv} does not exist.")
        return

    df = pd.read_csv(improvements_csv)
    scenarios = df["scenario"].tolist()
    labels = [s.capitalize() for s in scenarios]

    wait_reduction = -df["wait_time_change_pct"]  # Positive means reduction (good)
    queue_reduction = -df["queue_len_change_pct"]
    delay_reduction = -df["total_delay_change_pct"]
    thr_gain = df["throughput_change_pct"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    x = np.arange(len(scenarios))
    width = 0.2

    ax.bar(x - 1.5 * width, wait_reduction, width, label="Waiting Time Reduction (%)", color="#2ecc71")
    ax.bar(x - 0.5 * width, queue_reduction, width, label="Queue Length Reduction (%)", color="#3498db")
    ax.bar(x + 0.5 * width, delay_reduction, width, label="Total Delay Reduction (%)", color="#1abc9c")
    ax.bar(x + 1.5 * width, thr_gain, width, label="Throughput Gain (%)", color="#f39c12")

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Improvement Relative to Fixed-Time Baseline (%)")
    ax.set_title("Summary of DQN Performance Improvements Across Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved overall summary plot: {output_path}")


def plot_traffic_dynamics_timeseries(
    traces_path: str = "results/simulations/simulation_traces.json",
    scenario: str = "changing",
    output_path: str = "results/figures/traffic_dynamics_timeseries.png",
) -> None:
    """Plot time-series evolution of queues comparing Fixed-Time vs DQN."""
    if not os.path.exists(traces_path):
        print(f"[WARNING] Traces file {traces_path} does not exist.")
        return

    with open(traces_path, "r") as f:
        traces = json.load(f)

    if scenario not in traces or "fixed_time" not in traces[scenario] or "dqn" not in traces[scenario]:
        print(f"[WARNING] Scenario {scenario} missing traces in {traces_path}.")
        return

    ft_trace = traces[scenario]["fixed_time"]
    dqn_trace = traces[scenario]["dqn"]

    time_mins_ft = [t / 60.0 for t in ft_trace["time"]]
    time_mins_dqn = [t / 60.0 for t in dqn_trace["time"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=200)

    # Fixed-Time Subplot
    ax1.plot(time_mins_ft, ft_trace["ns_queue"], color="#3498db", label="North-South Queue", linewidth=1.8)
    ax1.plot(time_mins_ft, ft_trace["ew_queue"], color="#9b59b6", label="East-West Queue", linewidth=1.8)
    ax1.set_title("Fixed-Time Controller Queue Progression (Severe Congestion Spikes During Surges)")
    ax1.set_ylabel("Queued Vehicles")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # DQN Subplot
    ax2.plot(time_mins_dqn, dqn_trace["ns_queue"], color="#3498db", label="North-South Queue", linewidth=1.8)
    ax2.plot(time_mins_dqn, dqn_trace["ew_queue"], color="#9b59b6", label="East-West Queue", linewidth=1.8)
    ax2.set_title("DQN Controller Queue Progression (Dynamic Phase Adaptation Prevents Congestion Collapse)")
    ax2.set_xlabel("Simulation Elapsed Time (minutes)")
    ax2.set_ylabel("Queued Vehicles")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved traffic dynamics plot: {output_path}")


def plot_phase_timeline(
    traces_path: str = "results/simulations/simulation_traces.json",
    scenario: str = "medium",
    output_path: str = "results/figures/phase_timeline.png",
    duration_window: float = 300.0,  # 5-minute zoom
) -> None:
    """Plot Gantt-style signal phase allocations comparing Fixed-Time vs DQN."""
    if not os.path.exists(traces_path):
        return

    with open(traces_path, "r") as f:
        traces = json.load(f)

    if scenario not in traces:
        return

    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)

    for idx, c_key in enumerate(["fixed_time", "dqn"]):
        trace = traces[scenario][c_key]
        times = trace["time"]
        phases = trace["phase"]

        # Filter window
        win_times = [t for t in times if t <= duration_window]
        win_phases = [phases[i] for i, t in enumerate(times) if t <= duration_window]

        y_base = idx * 2.5
        for i in range(len(win_times) - 1):
            t_start = win_times[i]
            t_end = win_times[i + 1]
            p = win_phases[i]

            if p == 0:  # NS_GREEN
                color = "#2ecc71"
            elif p == 1:  # NS_CLEARANCE (Yellow)
                color = "#f1c40f"
            elif p == 2:  # EW_GREEN
                color = "#3498db"
            elif p == 3:  # EW_CLEARANCE (Yellow)
                color = "#e67e22"
            else:
                color = "#95a5a6"

            ax.barh(y_base, t_end - t_start, left=t_start, height=1.2, color=color, edgecolor="none")

    ax.set_yticks([0.0, 2.5])
    ax.set_yticklabels(["Fixed-Time Baseline\n(Rigid 30s/30s)", "DQN Controller\n(Demand-Responsive)"])
    ax.set_xlabel("Time into Simulation (seconds)")
    ax.set_xlim(0, duration_window)
    ax.set_title("Signal Phase Allocation Timeline (First 5 Minutes)")

    # Legend elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="NS Green Phase"),
        Patch(facecolor="#f1c40f", label="NS Yellow / Clearance"),
        Patch(facecolor="#3498db", label="EW Green Phase"),
        Patch(facecolor="#e67e22", label="EW Yellow / Clearance"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved phase timeline plot: {output_path}")


def plot_intersection_diagram(
    output_path: str = "results/figures/intersection_diagram.png",
) -> None:
    """Generate high-resolution intersection schematic diagram."""
    fig, ax = plt.subplots(figsize=(9, 9), dpi=200)

    # Road background
    ax.fill_between([-10, 10], -50, 50, color="#34495e")  # North-South road
    ax.fill_between([-50, 50], -10, 10, color="#34495e")  # East-West road

    # Intersection box
    ax.add_patch(patches.Rectangle((-10, -10), 20, 20, color="#2c3e50"))

    # Lane markings
    ax.plot([0, 0], [10, 50], color="#f1c40f", linestyle="--", linewidth=2)
    ax.plot([0, 0], [-50, -10], color="#f1c40f", linestyle="--", linewidth=2)
    ax.plot([10, 50], [0, 0], color="#f1c40f", linestyle="--", linewidth=2)
    ax.plot([-50, -10], [0, 0], color="#f1c40f", linestyle="--", linewidth=2)

    # Stop lines
    ax.plot([-10, 0], [10, 10], color="white", linewidth=4)
    ax.plot([0, 10], [-10, -10], color="white", linewidth=4)
    ax.plot([10, 10], [0, 10], color="white", linewidth=4)
    ax.plot([-10, -10], [-10, 0], color="white", linewidth=4)

    # Signal heads (Traffic lights)
    # North incoming
    ax.scatter([-5], [12], color="#2ecc71", s=180, edgecolors="white", linewidth=2, label="Green Signal (Active Phase)")
    # South incoming
    ax.scatter([5], [-12], color="#2ecc71", s=180, edgecolors="white", linewidth=2)
    # East incoming
    ax.scatter([12], [5], color="#e74c3c", s=180, edgecolors="white", linewidth=2, label="Red Signal (Conflicting Phase)")
    # West incoming
    ax.scatter([-12], [-5], color="#e74c3c", s=180, edgecolors="white", linewidth=2)

    # Approach labels & Queue indicator zones
    ax.annotate("NORTH APPROACH\nQueue Q_N (Inflow λ_N)", xy=(-5, 30), ha="center", color="white", fontweight="bold", fontsize=10)
    ax.annotate("SOUTH APPROACH\nQueue Q_S (Inflow λ_S)", xy=(5, -30), ha="center", color="white", fontweight="bold", fontsize=10)
    ax.annotate("EAST APPROACH\nQueue Q_E (Inflow λ_E)", xy=(30, 5), ha="center", color="white", fontweight="bold", fontsize=10)
    ax.annotate("WEST APPROACH\nQueue Q_W (Inflow λ_W)", xy=(-30, -5), ha="center", color="white", fontweight="bold", fontsize=10)

    # Movement arrows
    ax.arrow(-5, 45, 0, -25, head_width=2.5, head_length=3, fc="#ecf0f1", ec="#ecf0f1")
    ax.arrow(5, -45, 0, 25, head_width=2.5, head_length=3, fc="#ecf0f1", ec="#ecf0f1")
    ax.arrow(45, 5, -25, 0, head_width=2.5, head_length=3, fc="#ecf0f1", ec="#ecf0f1")
    ax.arrow(-45, -5, 25, 0, head_width=2.5, head_length=3, fc="#ecf0f1", ec="#ecf0f1")

    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Simulated 4-Way Signalized Intersection Architecture", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved intersection schematic: {output_path}")


def generate_all_visualizations():
    """Batch generate all mandatory visualizations."""
    os.makedirs("results/figures", exist_ok=True)
    plot_training_curves()
    plot_evaluation_metrics()
    plot_overall_summary()
    plot_traffic_dynamics_timeseries()
    plot_phase_timeline()
    plot_intersection_diagram()


if __name__ == "__main__":
    generate_all_visualizations()
