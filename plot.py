from typing import List
import numpy as np
import re
import os
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

FONT_SIZE = 22
TICK_LEN = 10
TICK_WDT = 3


def ema_smooth(arr, alpha=0.2):
    """Apply exponential moving average to 1D array."""
    smoothed = np.zeros_like(arr, dtype=float)
    smoothed[0] = arr[0]
    for t in range(1, len(arr)):
        smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * arr[t]
    return smoothed


def hex_to_rgb_tuple_str(hex_color):
    hex_color = hex_color.lstrip("#")
    return ",".join(str(int(hex_color[i : i + 2], 16)) for i in (0, 2, 4))


def plot_combined_cmorl_rover(
    base_source: str,
    title: str,
    output_name: str,
    start_safety: int = None,
    pretraining: int = None,
):
    pio.templates.default = "plotly_dark"

    results_dir = "results/rover"
    pattern = re.compile(rf"{base_source}(\d+(?:\.\d+)?).npz")

    reward_list, sat_list = [], []
    cost_avoid_list, cost_charger_list, cost_battery_list, all_costs = [], [], [], []
    lidar_list, battery_list, charger_time_list = [], [], []
    rho_goal_list, rho_avoid_list, rho_charger_list, rho_battery_list = [], [], [], []
    goal_list, collision_list, battery_end_list, truncated_list = [], [], [], []

    for fname in os.listdir(results_dir):
        match = pattern.match(fname)
        if match:
            print(f"Loading: {fname}")
            path = os.path.join(results_dir, fname)
            data = np.load(path)

            reward_list.append(data["rewards"])

            cost_avoid_list.append(data["cost_avoid"])
            cost_charger_list.append(data["cost_charger"])
            cost_battery_list.append(data["cost_battery"])
            if "all_costs" in data.keys():
                all_costs.append((data["all_costs"] * 100) - 50)

            lidar_list.append(data["mean_lidar"])
            battery_list.append(data["mean_battery"])
            charger_time_list.append(data["mean_charger_time"])

            rho_goal_list.append(data["rho_goal"])
            rho_avoid_list.append(data["rho_avoid"])
            rho_charger_list.append(data["rho_charger"])
            rho_battery_list.append(data["rho_battery"])

            goal_list.append(data["goal"] * 100)
            collision_list.append(data["collision"] * 100)
            battery_end_list.append(data["battery"] * 100)
            truncated_list.append(data["truncated"] * 100)

    # Convert to arrays
    reward_arr = np.array(reward_list)
    sat_arr = np.array(sat_list)
    cost_arr = np.array(all_costs)

    mean_r, std_r = reward_arr.mean(axis=0), reward_arr.std(axis=0)
    mean_c, std_c = cost_arr.mean(axis=0), cost_arr.std(axis=0)
    steps = np.arange(reward_arr.shape[1])

    # Event outcomes (row 1, col 3)

    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[[{"colspan": 2}, None, {"colspan": 2}, None], [{}, {}, {}, {}]],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
        subplot_titles=(
            "Reward & Cost Over Episodes",
            "",
            # "Episode Outcomes (Goal, Collision, Battery, Truncated)",
            # "",
            "Mean minimum lidar distance",
            "Mean Battery",
            "Mean steps spent at the charger",
            "Rho (Avoid, Charger Time)",
        ),
    )

    def add_outcome_trace(data_list, label, color_hex, col=3):
        data = np.array(data_list)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        rgb_str = hex_to_rgb_tuple_str(color_hex)

        # Shaded region
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})"),
            ),
            row=1,
            col=col,
        )

    # Top row: outcomes (goal/collision/battery/truncated)
    add_outcome_trace(goal_list, "Goal", "#2ca02c")
    add_outcome_trace(collision_list, "Collision", "#d62728")
    add_outcome_trace(battery_end_list, "Battery End", "#9467bd")

    # Top row: reward/cost
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean_r - std_r, (mean_r + std_r)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_r,
            mode="lines",
            name="Reward",
            line=dict(color="rgb(0,100,80)"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean_c - std_c, (mean_c + std_c)[::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_c,
            mode="lines",
            name="Total Cost",
            line=dict(color="rgb(31, 119, 180)"),
        ),
        row=1,
        col=1,
    )

    # Top row: outcomes (goal/collision/battery/truncated)
    # Helper to add traces with shaded area
    def add_trace(data_list, label, color_hex, col):
        if not data_list:
            return None
        data = np.array(data_list)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        rgb_str = hex_to_rgb_tuple_str(color_hex)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})"),
            ),
            row=2,
            col=col,
        )
        return np.min(mean - std), np.max(mean + std)

    # Bottom row: mean values
    ranges = {
        1: add_trace(lidar_list, "Mean minimum lidar", "#e377c2", col=1),
        2: add_trace(battery_list, "Mean Battery", "#17becf", col=2),
        3: add_trace(charger_time_list, "Mean charger time", "#161dd2", col=3),
    }

    # Formula thresholds
    fig.add_hline(
        y=0.2,
        line=dict(color="red"),
        annotation_text="G( min_lidar > 0.2 )",
        annotation_position="top left",
        annotation_font_color="red",
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0.6,
        line=dict(color="red"),
        annotation_text="G( stay_at_charger U battery > 0.6 )",
        annotation_position="top left",
        annotation_font_color="red",
        row=2,
        col=2,
    )

    # Rho plots
    r2 = add_trace(rho_avoid_list, "Rho Avoid", "#ff7f0e", col=4)
    r3 = add_trace(rho_charger_list, "Rho Charger", "#1f77b4", col=4)
    r_vals = [r for r in [r2, r3] if r]
    ranges[4] = (
        (min(r[0] for r in r_vals), max(r[1] for r in r_vals)) if r_vals else None
    )

    # Apply y-axis ranges
    for i in range(1, 5):
        if ranges.get(i):
            fig.update_yaxes(range=ranges[i], row=2, col=i)

    # Add vertical lines for pretraining / start_safety
    def add_vline(xpos, name, color):
        for r in range(1, 3):
            for c in range(1, 5):
                try:
                    fig.add_vline(
                        x=xpos,
                        line=dict(color=color, dash="dash"),
                        annotation_font_color=color,
                        annotation_text=name,
                        annotation_position="bottom right",
                        row=r,
                        col=c,
                    )
                except ValueError as e:
                    pass

    if pretraining is not None:
        add_vline(pretraining, "Pretraining", "yellow")
    if start_safety is not None:
        add_vline(start_safety, "Safety", "green")

    # Layout
    fig.update_layout(
        height=1000,
        width=1800,
        title=title,
        font=dict(family="Helvetica, sans-serif", size=16, color="white"),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        xaxis_title="Episodes",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="#1e1e1e",
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    # Save image
    pio.write_image(
        fig,
        f"results/rover/media/{output_name}.png",
        format="png",
        width=1800,
        height=1000,
    )


def plot_combined_cmorl_rover_2(
    base_source: str,
    title: str,
    output_name: str,
    start_safety: int = None,
    pretraining: int = None,
):
    pio.templates.default = "plotly_dark"

    results_dir = "results/rover"
    pattern = re.compile(rf"{base_source}(\d+(?:\.\d+)?).npz")

    reward_list, sat_list = [], []
    cost_avoid_list, cost_charger_list, cost_battery_list, all_costs = [], [], [], []
    lidar_list, battery_list, charger_time_list = [], [], []
    rho_goal_list, rho_avoid_list, rho_charger_list, rho_battery_list = [], [], [], []
    goal_list, collision_list, battery_end_list, truncated_list = [], [], [], []

    dist_low_list, dist_high_list = [], []  # NEW

    for fname in os.listdir(results_dir):
        match = pattern.match(fname)
        if match:
            print(f"Loading: {fname}")
            path = os.path.join(results_dir, fname)
            data = np.load(path)

            reward_list.append(data["rewards"])

            cost_avoid_list.append(data["cost_avoid"])
            cost_charger_list.append(data["cost_charger"])
            cost_battery_list.append(data["cost_battery"])
            if "all_costs" in data.keys():
                all_costs.append((data["all_costs"] * 100) - 50)

            lidar_list.append(data["mean_lidar"])
            battery_list.append(data["mean_battery"])
            charger_time_list.append(data["mean_charger_time"])

            rho_goal_list.append(data["rho_goal"])
            rho_avoid_list.append(data["rho_avoid"])
            rho_charger_list.append(data["rho_charger"])
            rho_battery_list.append(data["rho_battery"])

            dist_low_list.append(data["all_mean_dist_low"])  # NEW
            dist_high_list.append(data["all_mean_dist_high"])  # NEW

            goal_list.append(data["goal"] * 100)
            collision_list.append(data["collision"] * 100)
            battery_end_list.append(data["battery"] * 100)
            truncated_list.append(data["truncated"] * 100)

    # Convert to arrays
    reward_arr = np.array(reward_list)
    sat_arr = np.array(sat_list)
    cost_arr = np.array(all_costs)

    mean_r, std_r = reward_arr.mean(axis=0), reward_arr.std(axis=0)
    mean_c, std_c = cost_arr.mean(axis=0), cost_arr.std(axis=0)
    steps = np.arange(reward_arr.shape[1])

    # Subplots (now 2x5 grid)
    fig = make_subplots(
        rows=2,
        cols=5,
        specs=[
            [{"colspan": 2}, None, {"colspan": 2}, None, None],
            [{}, {}, {}, {}, {}],
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
        subplot_titles=(
            "Reward & Cost Over Episodes",
            "",
            "Mean minimum lidar distance",
            "Mean Battery",
            "Mean steps spent at the charger",
            "Rho (Avoid, Charger Time)",
            "Mean Dist charger when low battery",  # NEW TITLE
        ),
    )

    # -------------------------
    # Helper functions
    # -------------------------
    def add_outcome_trace(data_list, label, color_hex, col=3):
        data = np.array(data_list)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        rgb_str = hex_to_rgb_tuple_str(color_hex)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})"),
            ),
            row=1,
            col=col,
        )

    def add_trace(data_list, label, color_hex, col):
        if not data_list:
            return None
        data = np.array(data_list)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        rgb_str = hex_to_rgb_tuple_str(color_hex)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})"),
            ),
            row=2,
            col=col,
        )
        return np.min(mean - std), np.max(mean + std)

    # -------------------------
    # Top row: reward & cost
    # -------------------------
    add_outcome_trace(goal_list, "Goal", "#2ca02c")
    add_outcome_trace(collision_list, "Collision", "#d62728")
    add_outcome_trace(battery_end_list, "Battery End", "#9467bd")

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean_r - std_r, (mean_r + std_r)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_r,
            mode="lines",
            name="Reward",
            line=dict(color="rgb(0,100,80)"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean_c - std_c, (mean_c + std_c)[::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_c,
            mode="lines",
            name="Total Cost",
            line=dict(color="rgb(31, 119, 180)"),
        ),
        row=1,
        col=1,
    )

    # -------------------------
    # Bottom row: metrics
    # -------------------------
    ranges = {
        1: add_trace(lidar_list, "Mean minimum lidar", "#e377c2", col=1),
        2: add_trace(battery_list, "Mean Battery", "#17becf", col=2),
        3: add_trace(charger_time_list, "Mean charger time", "#161dd2", col=3),
    }

    # Formula thresholds
    fig.add_hline(
        y=0.2,
        line=dict(color="red"),
        annotation_text="G( min_lidar > 0.2 )",
        annotation_position="top left",
        annotation_font_color="red",
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0.6,
        line=dict(color="red"),
        annotation_text="G( stay_at_charger U battery > 0.6 )",
        annotation_position="top left",
        annotation_font_color="red",
        row=2,
        col=2,
    )

    # Rho plots
    r2 = add_trace(rho_avoid_list, "Rho Avoid", "#ff7f0e", col=4)
    r3 = add_trace(rho_charger_list, "Rho Charger", "#1f77b4", col=4)
    r_vals = [r for r in [r2, r3] if r]
    ranges[4] = (
        (min(r[0] for r in r_vals), max(r[1] for r in r_vals)) if r_vals else None
    )

    # NEW: dist_low & dist_high
    d1 = add_trace(dist_low_list, "Mean Dist Low", "#bcbd22", col=5)
    d_vals = [d for d in [d1] if d]
    ranges[5] = (
        (min(d[0] for d in d_vals), max(d[1] for d in d_vals)) if d_vals else None
    )

    # Apply y-axis ranges
    for i in range(1, 6):
        if ranges.get(i):
            fig.update_yaxes(range=ranges[i], row=2, col=i)

    # -------------------------
    # Vertical lines
    # -------------------------
    def add_vline(xpos, name, color):
        for r in range(1, 3):
            for c in range(1, 6):
                try:
                    fig.add_vline(
                        x=xpos,
                        line=dict(color=color, dash="dash"),
                        annotation_font_color=color,
                        annotation_text=name,
                        annotation_position="bottom right",
                        row=r,
                        col=c,
                    )
                except ValueError:
                    pass

    if pretraining is not None:
        add_vline(pretraining, "Pretraining", "yellow")
    if start_safety is not None:
        add_vline(start_safety, "Safety", "green")

    # Layout
    fig.update_layout(
        height=1000,
        width=2200,
        title=title,
        font=dict(family="Helvetica, sans-serif", size=16, color="white"),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        xaxis_title="Episodes",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="#1e1e1e",
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    # Save image
    pio.write_image(
        fig,
        f"results/rover/media/{output_name}.png",
        format="png",
        width=2200,
        height=1000,
    )


def plot_multi_source_comparison_rover(
    title_prefix: str,
    output_prefix: str,
    start_safety: int = None,
):
    from plotly.subplots import make_subplots

    pio.templates.default = "simple_white"
    results_dir = "results/rover"

    base_sources = ["no_safety_", "stlgym_", "random_safety_70_", "our_safety_70_"]
    labels = ["No safety", "Reward shaping", "CMORL original", "Our"]

    # Collect metrics
    source_data = {}
    for base_source in base_sources:
        pattern = re.compile(rf"{base_source}(\d+(?:\.\d+)?).npz")
        reward_list, cost_list, lidar_list, battery_list, charger_time_list = (
            [],
            [],
            [],
            [],
            [],
        )
        rho_avoid_list, rho_charger_list, rho_battery_list, rho_goal_list = (
            [],
            [],
            [],
            [],
        )
        goal_list, collision_list, battery_end_list, truncated_list = [], [], [], []

        for fname in os.listdir(results_dir):
            if pattern.match(fname):
                path = os.path.join(results_dir, fname)
                data = np.load(path)
                reward_list.append(ema_smooth(data["rewards"], 0.9))
                cost_list.append(
                    ema_smooth(data["all_costs"] * 100 - 50, 0.9)
                    if "all_costs" in data
                    else None
                )
                lidar_list.append(ema_smooth(data["mean_lidar"], 0.7))
                battery_list.append(ema_smooth(data["mean_battery"], 0.7))
                charger_time_list.append(ema_smooth(data["mean_charger_time"], 0.7))
                rho_avoid_list.append(ema_smooth(data["rho_avoid"], 0.7))
                rho_charger_list.append(ema_smooth(data["rho_charger"], 0.7))
                rho_battery_list.append(ema_smooth(data["rho_battery"], 0.7))
                rho_goal_list.append(ema_smooth(data["rho_goal"], 0.7))
                goal_list.append(data["goal"] * 100)
                collision_list.append(data["collision"] * 100)
                battery_end_list.append(data["battery"] * 100)
                truncated_list.append(data["truncated"] * 100)

        # Avoid ambiguous truth value with arrays
        cost_array = (
            np.array([c for c in cost_list if c is not None])
            if any(c is not None for c in cost_list)
            else None
        )

        source_data[base_source] = {
            "reward": np.array(reward_list),
            "cost": cost_array,
            "lidar": np.array(lidar_list),
            "battery": np.array(battery_list),
            "charger_time": np.array(charger_time_list),
            "rho_avoid": np.array(rho_avoid_list),
            "rho_charger": np.array(rho_charger_list),
            "rho_battery": np.array(rho_battery_list),
            "rho_goal": np.array(rho_goal_list),
            "goal_list": np.array(goal_list),
            "collision_list": np.array(collision_list),
            "battery_end_list": np.array(battery_end_list),
            "truncated_list": np.array(truncated_list),
        }

    metrics = {
        "reward": "Reward",
        "cost": "Cost",
        "lidar": "Mean Lidar",
        "battery": "Mean Battery",
        "charger_time": "Charger Time",
        "rho_avoid": "Rho Avoid",
        "rho_charger": "Rho Charger",
        "rho_battery": "Rho Battery",
        "rho_goal": "Rho Goal",
    }

    palette = ["#1f77b4", "#6B4F3A", "#ff7f0e", "#D81B60", "#9467bd", "#8c564b"]

    steps = np.arange(next(iter(source_data.values()))["reward"].shape[1])

    def hex_to_rgb_tuple_str(hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        return ",".join(str(int(hex_color[i : i + 2], 16)) for i in (0, 2, 4))

    def add_metric_trace(fig, arr, label, color, highlight=False, showlegend=True):
        if arr is None or len(arr) == 0:
            return
        mean = arr.mean(axis=0)
        n = arr.shape[0]
        sem = arr.std(axis=0) / np.sqrt(n)
        std = 1.96 * sem
        rgb_str = hex_to_rgb_tuple_str(color)
        opacity = 0.25 if highlight else 0.1
        width = 2 if highlight else 1
        dash = "solid"

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},{opacity})",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})", width=width, dash=dash),
                showlegend=showlegend,
                legendrank=0 if highlight else 1,
            )
        )

    # Plot metrics
    rho_metrics = ["rho_avoid", "rho_charger", "lidar", "battery", "charger_time"]
    for metric, title in metrics.items():
        print(metric)
        if metric in rho_metrics:
            fig = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.1)

            subplot_groups = [
                (["Reward shaping", "Our"], 1),
                (["No safety", "CMORL original"], 2),
            ]

            all_data = []
            for group_labels, row in subplot_groups:
                
                
                for i, (base_source, label) in enumerate(zip(base_sources, labels)):
                    if label not in group_labels:
                        continue
                    color = palette[i % len(palette)]
                    data = source_data[base_source]
                    highlight = label.lower() == "our"

                    mean = data[metric].mean(axis=0)
                    n = data[metric].shape[0]
                    sem = data[metric].std(axis=0) / np.sqrt(n)
                    std = 1.96 * sem
                    
                    all_data.extend(np.array([mean + std, mean - std]))

                    rgb_str = hex_to_rgb_tuple_str(color)

                    # Shaded standard deviation
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([steps, steps[::-1]]),
                            y=np.concatenate([mean - std, (mean + std)[::-1]]),
                            fill="toself",
                            fillcolor=f"rgba({rgb_str},0.1)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=1,
                        col=row,
                    )

                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=steps,
                            y=mean,
                            mode="lines",
                            name=label,
                            line=dict(
                                color=f"rgb({rgb_str})",
                                width=2 if highlight else 1,
                                dash="solid",
                            ),
                            showlegend=True,  # only show legend once
                            legendrank=0 if highlight else 1,
                        ),
                        row=1,
                        col=row,
                    )

            # Vertical safety marker
            if start_safety is not None:
                fig.add_vline(
                    x=start_safety,
                    line=dict(
                        color="green",
                        dash="dash",
                        width=2,
                    ),
                    annotation_text="start safety",
                    annotation_position="bottom right",
                    annotation_font_color="green",
                    annotation_ax=30,
                )

            # rg = [-0.4, 0.2] if metric == "rho_charger" else [-0.1, 0.05]

            y_axis_title = {
                "rho_avoid": "Robustness",
                "rho_charger": "Robustness",
                "lidar": "Lidar",
                "battery": "Battery",
                "charger_time": "Charger time",
            }

            fig.update_layout(
                # title=f"{title_prefix} - {title}",
                font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
                xaxis_title="Timesteps 5 × 10<sup>4</sup>",
                yaxis_title=y_axis_title[metric],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                ),
                # yaxis=dict(range=rg),
                # yaxis2=dict(range=rg),
                # yaxis3=dict(range=rg),
                # yaxis4=dict(range=rg),
            )
            
            fig.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            fig.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            
            all_data = np.array(all_data).flatten()
            # y_min, y_max = np.percentile(all_data, [1, 99])
            y_min, y_max = all_data.min(), all_data.max()
            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
            fig.update_yaxes(range=[y_min, y_max], row=1, col=2)
            
            all_data = []

            # Save figure
            pio.write_image(
                fig,
                f"results/rover/media/{output_prefix}_{metric}_subplots.png",
                format="png",
                width=1200,
                height=500,
                scale=2,
            )
        else:
            fig = go.Figure()
            for j, (base_source, label) in enumerate(zip(base_sources, labels)):
                color = palette[j % len(palette)]
                data = source_data[base_source]
                highlight = label == "Our"
                add_metric_trace(
                    fig,
                    data[metric],
                    label,
                    color,
                    highlight=highlight,
                    showlegend=True,
                )

            if start_safety is not None:
                fig.add_vline(
                    x=start_safety,
                    line=dict(
                        color="green",
                        dash="dash",
                        width=2,
                    ),
                    annotation_text="start safety",
                    annotation_position="bottom right",
                    annotation_font_color="green",
                    annotation_ax=30,
                )
                
            y_axis_title = {
                "cost": "Total cost",
                "reward": "Return",
                "battery": "Battery",
                "rho_battery": "Robustness",
                "rho_goal": "Robustness",
            }

            fig.update_layout(
                height=600,
                width=1000,
                # title=f"{title_prefix} - {title}",
                font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
                xaxis_title="Timesteps 5 × 10<sup>4</sup>",
                yaxis_title=y_axis_title[metric],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                ),
            )
            
           
            
            fig.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            fig.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            
            pio.write_image(
                fig,
                f"results/rover/media/{output_prefix}_{metric}.png",
                format="png",
                width=1200,
                height=700,
                scale=2,
            )

    our_source = base_sources[0]
    our_label = labels[0]
    our_data = source_data[our_source]
    color = palette[0]

    def maybe_add_safety_marker(fig):
        if start_safety is not None:
            fig.add_shape(
                type="line",
                x0=start_safety,
                x1=start_safety,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="green", dash="dash", width=2),
            )

    # 1. Goal vs Collision vs Battery End
    fig = go.Figure()
    add_metric_trace(
        fig, our_data["goal_list"], f"{our_label} - Goal %", color, showlegend=True
    )
    add_metric_trace(
        fig,
        our_data["collision_list"],
        f"{our_label} - Collision %",
        color,
        showlegend=True,
    )
    add_metric_trace(
        fig,
        our_data["battery_end_list"],
        f"{our_label} - Battery End %",
        color,
        showlegend=True,
    )
    maybe_add_safety_marker(fig)

    fig.update_layout(
        height=600,
        width=1000,
        title=f"{title_prefix} - Goal / Collision / Battery End ({our_label})",
        font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
        xaxis_title="Batch",
        yaxis_title="Percentage",
    )
    pio.write_image(
        fig,
        f"results/rover/media/{output_prefix}_our_goal_collision_battery.png",
        format="png",
        width=1200,
        height=700,
        scale=2,
    )

    # # 2. Lidar
    # fig = go.Figure()
    # add_metric_trace(
    #     fig,
    #     our_data["lidar"],
    #     f"{our_label} - Lidar",
    #     color,
    #     showlegend=False,
    #     highlight=True,
    # )
    # maybe_add_safety_marker(fig)
    # fig.update_layout(
    #     height=600,
    #     width=1000,
    #     title=f"{title_prefix} - Lidar ({our_label})",
    #     font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
    #     xaxis_title="Batch",
    # )
    # pio.write_image(
    #     fig,
    #     f"results/rover/media/{output_prefix}_our_lidar.png",
    #     format="png",
    #     width=1200,
    #     height=700,
    #     scale=2,
    # )

    # # 3. Battery
    # fig = go.Figure()
    # add_metric_trace(
    #     fig,
    #     our_data["battery"],
    #     f"{our_label} - Battery",
    #     color,
    #     showlegend=False,
    #     highlight=True,
    # )
    # maybe_add_safety_marker(fig)
    # fig.update_layout(
    #     height=600,
    #     width=1000,
    #     title=f"{title_prefix} - Battery ({our_label})",
    #     font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
    #     xaxis_title="Batch",
    # )
    # pio.write_image(
    #     fig,
    #     f"results/rover/media/{output_prefix}_our_battery.png",
    #     format="png",
    #     width=1200,
    #     height=700,
    #     scale=2,
    # )

    # # 4. Charger Time
    # fig = go.Figure()
    # add_metric_trace(
    #     fig,
    #     our_data["charger_time"],
    #     f"{our_label} - Charger Time",
    #     color,
    #     showlegend=False,
    #     highlight=True,
    # )
    # maybe_add_safety_marker(fig)
    # fig.update_layout(
    #     height=600,
    #     width=1000,
    #     title=f"{title_prefix} - Charger Time ({our_label})",
    #     font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
    #     xaxis_title="Batch",
    # )
    # pio.write_image(
    #     fig,
    #     f"results/rover/media/{output_prefix}_our_charger_time.png",
    #     format="png",
    #     width=1200,
    #     height=700,
    #     scale=2,
    # )


def plot_combined_cmorl_pendulum(
    base_source: str, title: str, output_name: str, start_safety: int = None
):
    pio.templates.default = "plotly_dark"

    results_dir = "results/pendulum"
    pattern = re.compile(rf"{base_source}(\d+(?:\.\d+)?).npz")

    reward_list, cost_list = [], []
    theta_list, thetadot_list, torque_list = [], [], []
    rho_thetadot_list, rho_torque_list = [], []

    for fname in os.listdir(results_dir):
        match = pattern.match(fname)
        if match:
            print(f"Loading: {fname}")
            path = os.path.join(results_dir, fname)
            data = np.load(path)

            reward_list.append(data["rewards"] / 10)
            theta_list.append(data["theta"])
            thetadot_list.append(data["thetadot"])
            torque_list.append(data["torque"])
            rho_thetadot_list.append(data["rho_thetadot"])
            rho_torque_list.append(data["rho_torque"])

            total_cost = -(data["cost_thetadot"] + data["cost_torque"])
            cost_list.append(total_cost)

    reward_arr, cost_arr = np.array(reward_list), np.array(cost_list)
    mean_r, std_r = reward_arr.mean(axis=0), reward_arr.std(axis=0)
    mean_c, std_c = cost_arr.mean(axis=0), cost_arr.std(axis=0)
    steps = np.arange(reward_arr.shape[1])

    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[[None, {"colspan": 2}, None, None], [{}, {}, {}, {}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.15,
        subplot_titles=(
            "Reward & Cost Over Episodes",
            "Theta",
            "ThetaDot",
            "Torque",
            "Rho (Thetadot + Torque)",
        ),
    )

    # Top row: reward/cost
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean_r - std_r, (mean_r + std_r)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_r,
            mode="lines",
            name="Reward",
            line=dict(color="rgb(0,100,80)"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean_c - std_c, (mean_c + std_c)[::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_c,
            mode="lines",
            name="Cost",
            line=dict(color="rgb(31, 119, 180)"),
        ),
        row=1,
        col=2,
    )

    if start_safety is not None:
        fig.add_vline(
            x=start_safety,
            line=dict(color="green", dash="dash"),
            row=1,
            col=2,
            annotation_text="start safety",
            annotation_position="top right",
            annotation_font_color="green",
        )

    # Helper to add traces with shaded area
    def add_trace(data_list, label, color_hex, col):
        if not data_list:
            return None
        data = np.array(data_list)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        rgb_str = hex_to_rgb_tuple_str(color_hex)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})"),
            ),
            row=2,
            col=col,
        )
        return np.min(mean - std), np.max(mean + std)

    # Bottom row: theta, thetadot, torque, rho
    ranges = {
        1: add_trace(np.abs(theta_list).tolist(), "Theta", "#e377c2", col=1),
        2: add_trace(np.abs(thetadot_list).tolist(), "Thetadot", "#17becf", col=2),
        3: add_trace(np.abs(torque_list).tolist(), "Torque", "#161dd2", col=3),
    }

    r1 = add_trace(
        rho_thetadot_list, "Rho F(G( abs(thetadot) < 0.5 ))", "#bcbd22", col=4
    )
    r2 = add_trace(rho_torque_list, "Rho F(G( abs(torque) < 0.3 ))", "#ff7f0e", col=4)
    if r1 and r2:
        ranges[4] = (min(r1[0], r2[0]), max(r1[1], r2[1]))
    elif r1:
        ranges[4] = r1
    elif r2:
        ranges[4] = r2
    else:
        ranges[4] = None

    # Add reference lines to bottom row
    if start_safety is not None:
        for col in [2, 3]:
            fig.add_vline(
                x=start_safety,
                line=dict(color="green", dash="dash"),
                row=2,
                col=col,
                annotation_text="start safety",
                annotation_position="top right",
                annotation_font_color="green",
            )

    fig.add_hline(
        y=0.5,
        line=dict(color="red"),
        annotation_text="F(G( abs(thetadot) < 0.5 ))",
        annotation_position="top right",
        annotation_font_color="red",
        row=2,
        col=2,
    )
    fig.add_hline(
        y=0.3,
        line=dict(color="red"),
        annotation_text="F(G( abs(torque) < 0.3 ))",
        annotation_position="top right",
        annotation_font_color="red",
        row=2,
        col=3,
    )

    # Apply y-axis range if available
    for i in range(1, 5):
        if ranges.get(i):
            fig.update_yaxes(range=ranges[i], row=2, col=i)

    # Layout
    fig.update_layout(
        height=900,
        width=1600,
        title=title,
        font=dict(family="Helvetica, sans-serif", size=16, color="white"),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        xaxis_title="Episodes",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="#1e1e1e",
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    # Save image
    pio.write_image(
        fig,
        f"results/pendulum/media/{output_name}.png",
        format="png",
        width=1600,
        height=900,
    )


def plot_multi_source_comparison_pendulum(
    base_sources: List[str],
    labels: List[str],
    title_prefix: str,
    output_prefix: str,
    start_safety: int = None,
    show_legend: bool = True,
):
    pio.templates.default = "simple_white"
    results_dir = "results/pendulum"

    base_sources = [
        "no_safety_10000_",
        "stlgym_10000_",
        "random_safety_100_",
        "our_safety_100_",
    ]
    labels = ["No safety", "Reward shaping", "CMORL original", "Our"]

    # Collect metrics for each base source
    source_data = {}
    for base_source in base_sources:
        pattern = re.compile(rf"{base_source}(\d+(?:\.\d+)?).npz")

        reward_list, cost_list = [], []
        theta_list, thetadot_list, torque_list = [], [], []
        rho_thetadot_list, rho_torque_list = [], []

        for fname in os.listdir(results_dir):
            if pattern.match(fname):
                path = os.path.join(results_dir, fname)
                data = np.load(path)

                reward_list.append((data["rewards"] / 10))
                theta_list.append((np.abs(data["theta"])))
                thetadot_list.append((np.abs(data["thetadot"])))
                torque_list.append((np.abs(data["torque"])))
                rho_thetadot_list.append((data["rho_thetadot"]))
                rho_torque_list.append((data["rho_torque"]))

                total_cost = -(data["cost_thetadot"] + data["cost_torque"])
                cost_list.append(ema_smooth(total_cost))

        source_data[base_source] = {
            "reward": np.array(reward_list),
            "cost": np.array(cost_list),
            "theta": np.array(theta_list),
            "thetadot": np.array(thetadot_list),
            "torque": np.array(torque_list),
            "rho_thetadot": np.array(rho_thetadot_list),
            "rho_torque": np.array(rho_torque_list),
        }

    metrics = {
        "reward": "Reward",
        "cost": "Cost",
        "theta": "Theta",
        "thetadot": "ThetaDot",
        "torque": "Torque",
        "rho_thetadot": "Rho Thetadot",
        "rho_torque": "Rho Torque",
    }

    palette = ["#1f77b4", "#6B4F3A", "#ff7f0e", "#D81B60", "#9467bd", "#8c564b"]

    steps = np.arange(next(iter(source_data.values()))["reward"].shape[1])

    def add_metric_trace(fig, arr, label, color, highlight=False, showlegend=True):
        if arr is None or len(arr) == 0:
            return
        mean = arr.mean(axis=0)

        n = arr.shape[0]

        sem = arr.std(axis=0) / np.sqrt(n)
        std = 1.96 * sem

        # std = arr.std(axis=0)
        rgb_str = hex_to_rgb_tuple_str(color)

        # Adjust style depending on highlight
        opacity = 0.25 if highlight else 0.1
        width = 2 if highlight else 1
        dash = "solid"

        # Std shading
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},{opacity})",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendrank=0 if highlight else 1,
            )
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})", width=width, dash=dash),
                showlegend=showlegend,
                legendrank=0 if highlight else 1,
            )
        )

    # Plot one figure per metric
    for metric, title in metrics.items():
        if metric in ["rho_thetadot", "rho_torque", "thetadot", "torque"]:
            # One figure with two subplots
            fig = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.1)

            subplot_groups = [
                (["Reward shaping", "Our"], 1),
                (["No safety", "CMORL original"], 2),
            ]

            all_data = []
            for group_labels, row in subplot_groups:
                for i, (base_source, label) in enumerate(zip(base_sources, labels)):
                    if label not in group_labels:
                        continue
                    color = palette[i % len(palette)]
                    data = source_data[base_source]
                    highlight = label.lower() == "our"

                    mean = data[metric].mean(axis=0)
                    
                    n = data[metric].shape[0]
                    sem = data[metric].std(axis=0) / np.sqrt(n)
                    std = 1.96 * sem
                    
                    all_data.extend(np.array([mean + std, mean - std]))

                    rgb_str = hex_to_rgb_tuple_str(color)

                    # Shaded standard deviation
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([steps, steps[::-1]]),
                            y=np.concatenate([mean - std, (mean + std)[::-1]]),
                            fill="toself",
                            fillcolor=f"rgba({rgb_str},0.1)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=1,
                        col=row,
                    )

                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=steps,
                            y=mean,
                            mode="lines",
                            name=label,
                            line=dict(
                                color=f"rgb({rgb_str})",
                                width=2 if highlight else 1,
                                dash="solid",
                            ),
                            showlegend=show_legend,
                            legendrank=0 if highlight else 1,
                        ),
                        row=1,
                        col=row,
                    )

            # Vertical safety marker
            if start_safety is not None:
                fig.add_vline(
                    x=start_safety,
                    line=dict(
                        color="green",
                        dash="dash",
                        width=2,
                    ),
                    annotation_text="start safety",
                    annotation_position="bottom right",
                    annotation_font_color="green",
                    annotation_ax=30,
                )

            y_axis_title = {
                "rho_thetadot": "Robustness",
                "rho_torque": "Robustness",
                "thetadot": "Angular velocity",
                "torque": "Torque",
            }

            if metric == "thetadot":
                fig.add_hline(
                    y=0.5,
                    line=dict(
                        color="red",
                        width=3,
                    ),
                    annotation_text="0.5",
                    annotation_position="top right",
                    annotation_font_color="red",
                )

            if metric == "torque":
                fig.add_hline(
                    y=0.3,
                    line=dict(
                        color="red",
                        width=3,
                    ),
                    annotation_text="0.3",
                    annotation_position="top right",
                    annotation_font_color="red",
                )

            fig.update_layout(
                # title=f"{title_prefix} - {title}",
                font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
                xaxis_title="Timesteps 5 × 10<sup>4</sup>",
                yaxis_title=y_axis_title[metric],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                ),
            )
            
            
            fig.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            fig.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            
            all_data = np.array(all_data).flatten()
            # y_min, y_max = np.percentile(all_data, [1, 99])
            y_min, y_max = all_data.min(), all_data.max()
            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
            fig.update_yaxes(range=[y_min, y_max], row=1, col=2)
            
            all_data = []

            # Save figure
            pio.write_image(
                fig,
                f"results/pendulum/media/{output_prefix}_{metric}_subplots.png",
                format="png",
                width=1200,
                height=500,
                scale=2,
            )

        else:
            # Normal plotting for other metrics (keep your original code)
            fig = go.Figure()
            for i, (base_source, label) in enumerate(zip(base_sources, labels)):
                color = palette[i % len(palette)]
                data = source_data[base_source]
                highlight = label.lower() == "our"
                add_metric_trace(
                    fig,
                    data[metric],
                    label,
                    color,
                    highlight=highlight,
                    showlegend=True,
                )

            if start_safety is not None:
                fig.add_vline(
                    x=start_safety,
                    line=dict(
                        color="green",
                        dash="dash",
                        width=2,
                    ),
                    annotation_text="start safety",
                    annotation_position="bottom right",
                    annotation_font_color="green",
                    annotation_ax=30,
                )
                
            y_axis_title = {
                "cost": "Total cost",
                "reward": "Return",
                "theta": "Theta"
            }
                
            fig.update_layout(
                height=600,
                width=1000,
                # title=f"{title_prefix} - {title}",
                font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
                xaxis_title="Timesteps 5 × 10<sup>4</sup>",
                yaxis_title=y_axis_title[metric],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                ),
            )
            
            fig.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)

            fig.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
            
            pio.write_image(
                fig,
                f"results/pendulum/media/{output_prefix}_{metric}.png",
                format="png",
                width=1200,
                height=700,
                scale=2,
            )


def plot_ablation(
    base_source: str,
    title_prefix: str,
    output_prefix: str,
    results_dir: str = "results/pendulum",
    show_legend: bool = True,
):
    """
    Make one plot for reward, one for cost.
    Each start_safety value gets its own curve (averaged over seeds).
    """

    pio.templates.default = "simple_white"

    # Regex: base_source + safety + seed
    pattern = re.compile(rf"{base_source}(\d+)_(\d+)\.npz")

    # Collect metrics grouped by start_safety
    safety_data = {}  # {safety: {"reward": [..], "cost": [..]}}

    for fname in os.listdir(results_dir):
        match = pattern.match(fname)
        if match:
            start_safety, seed = int(match.group(1)), int(match.group(2))
            path = os.path.join(results_dir, fname)
            data = np.load(path)

            if "rover" in results_dir:
                reward = ema_smooth(data["rewards"], 0.9)
                cost = ema_smooth(data["all_costs"] * 100 - 50, 0.9)
            else:
                reward = data["rewards"] / 10
                total_cost = -(data["cost_thetadot"] + data["cost_torque"])
                cost = ema_smooth(total_cost)

            if start_safety not in safety_data:
                safety_data[start_safety] = {"reward": [], "cost": []}

            safety_data[start_safety]["reward"].append(reward)
            safety_data[start_safety]["cost"].append(cost)

    # Colors for different start_safety values
    palette = ["#1f77b4", "#D81B60", "#2ca02c", "#D81B60", "#9467bd", "#8c564b"]

    def add_metric_trace(fig, arr_list, label, color, showlegend=True):
        arr = np.array(arr_list)
        mean = arr.mean(axis=0)
        n = arr.shape[0]
        sem = arr.std(axis=0) / np.sqrt(n)
        ci95 = 1.96 * sem

        steps = np.arange(mean.shape[0])
        rgb_str = hex_to_rgb_tuple_str(color)

        # Std shading
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([mean - ci95, (mean + ci95)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb_str},0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=label,
                line=dict(color=f"rgb({rgb_str})", width=2),
                showlegend=showlegend,
            )
        )

    # Plot reward
    fig_reward = go.Figure()
    for i, (safety, metrics) in enumerate(sorted(safety_data.items())):
        color = palette[i % len(palette)]
        add_metric_trace(fig_reward, metrics["reward"], f"start_safety {safety}", color)

        fig_reward.add_vline(
            x=safety,
            line=dict(
                color=color,
                dash="dash",
                width=2,
            ),
        )

    fig_reward.update_layout(
        height=600,
        width=1000,
        font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
        xaxis_title="Timesteps 5 × 10<sup>4</sup>",
        yaxis_title="Return",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
        ),
    )
    
    fig_reward.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)

    fig_reward.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
    
    
    
    pio.write_image(
        fig_reward,
        f"{results_dir}/media/{output_prefix}_reward.png",
        format="png",
        width=1200,
        height=700,
        scale=2,
    )

    # Plot cost
    fig_cost = go.Figure()
    for i, (safety, metrics) in enumerate(sorted(safety_data.items())):
        color = palette[i % len(palette)]
        add_metric_trace(fig_cost, metrics["cost"], f"start_safety {safety}", color)

        fig_cost.add_vline(
            x=safety,
            line=dict(
                color=color,
                dash="dash",
                width=2,
            ),
        )

    fig_cost.update_layout(
        height=600,
        width=1000,
        font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
        xaxis_title="Timesteps 5 × 10<sup>4</sup>",
        yaxis_title="Total cost",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
        ),
    )
    
    fig_cost.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
    fig_cost.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
    
    pio.write_image(
        fig_cost,
        f"{results_dir}/media/{output_prefix}_cost.png",
        format="png",
        width=1200,
        height=700,
        scale=2,
    )


if __name__ == "__main__":
    # for start_safety in [40, 70, 100]:
    #     plot_combined_cmorl_rover(f"our_safety_{start_safety}_", f"CMORL On Rover, tuning with {start_safety}", f"cmorl_safety_{start_safety}", start_safety=start_safety)

    # start_safety = 70
    # plot_combined_cmorl_rover_2("no_safety_", "CMORL On Rover, No safety", "cmorl_no_safety")
    # plot_combined_cmorl_rover_2(f"our_safety_{start_safety}_", "CMORL On Rover, OUR", "cmorl_our", start_safety=start_safety)
    # plot_combined_cmorl_rover_2("stlgym_", "CMORL On Rover, STLGym", "cmorl_stlgym")
    # plot_combined_cmorl_rover_2(f"random_safety_{start_safety}_", "CMORL On Rover, Random critic", "cmorl_random", start_safety=start_safety)

    # for start_safety in [50, 100, 150]:
    #     plot_combined_cmorl_pendulum(f"our_safety_{start_safety}_", f"CMORL On Pendulum, tuning with {start_safety}", f"cmorl_safety_{start_safety}", start_safety=start_safety)

    # plot_combined_cmorl_pendulum("no_safety_10000_", "CMORL On Pendulum, No safety", "cmorl_no_safety")
    # plot_combined_cmorl_pendulum("stlgym_10000_", "CMORL On Pendulum, StlGym", "stlgym")
    # plot_combined_cmorl_pendulum(f"random_safety_{100}_", "CMORL On Pendulum, Random critic", "cmorl_random", start_safety=100)

    # All baselines + our

    plot_multi_source_comparison_rover(
        title_prefix="Rover ",
        output_prefix="rover",
        start_safety=70,
    )

    plot_multi_source_comparison_pendulum(
        base_sources=[
            "no_safety_10000_",
            "stlgym_10000_",
            "random_safety_100_",
            "our_safety_100_",
        ],
        labels=["No safety", "Reward shaping", "CMORL original", "Our"],
        title_prefix="Pendulum",
        output_prefix="pendulum",
        start_safety=100,
    )

    # Ablations

    plot_ablation(
        base_source="our_safety_",
        title_prefix="Pendulum Comparison",
        output_prefix="pendulum_ablation",
        results_dir="results/pendulum",
    )

    plot_ablation(
        base_source="our_safety_",
        title_prefix="Rover Comparison",
        output_prefix="rover_ablation",
        results_dir="results/rover",
    )
