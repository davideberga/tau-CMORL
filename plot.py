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
                
                # if  pattern.match(fname).group(1) in ['46']:
                #     continue
            
                
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
            
            # print(pattern.match(fname).group(2))
            # if pattern.match(fname).group(2) in ['44', '45', '46']:
            #     continue

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
