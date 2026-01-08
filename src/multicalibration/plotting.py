# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.subplots as sp

from multicalibration import methods, metrics, utils
from multicalibration.utils import BinningMethodInterface
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots


def _compute_calibration_curve(
    data: pd.DataFrame,
    score_col: str,
    label_col: str,
    num_bins: int,
    sample_weight_col: str | None = None,
    epsilon: float = 1e-6,
    binning_method: BinningMethodInterface = utils.make_equispaced_bins,
) -> pd.DataFrame:
    sample_weight = (
        np.ones_like(data[score_col].values)
        if sample_weight_col is None
        else data[sample_weight_col].values
    )

    bins = binning_method(data[score_col].values, num_bins, epsilon)
    label_prop_positive, lower, upper, bin_score_avg = utils.positive_label_proportion(
        labels=data[label_col].values,
        predictions=data[score_col].values,
        bins=bins,
        sample_weight=sample_weight,
    )
    return pd.DataFrame(
        {
            "label_prop_positive": label_prop_positive,
            "lower": lower,
            "upper": upper,
            "bin": bin_score_avg,
        }
    )


def plot_global_calibration_curve(
    data: pd.DataFrame,
    score_col: str,
    label_col: str,
    num_bins: int = metrics.CALIBRATION_ERROR_NUM_BINS,
    sample_weight_col: str | None = None,
    binning_method: str = "equispaced",
    plot_incomplete_cis: bool = True,
    x_lim: Tuple[float, float] = (0, 1.1),
) -> go.Figure:
    assert binning_method in ["equispaced", "equisized"]

    binning_fun = (
        utils.make_equispaced_bins
        if binning_method == "equispaced"
        else utils.make_equisized_bins
    )
    # TODO: Make confidence interval calculation for calibration curve with sample_weights more correct by calculating the effective sample size from the sample weights
    curves = _compute_calibration_curve(
        data,
        score_col=score_col,
        label_col=label_col,
        num_bins=num_bins,
        sample_weight_col=sample_weight_col,
        binning_method=binning_fun,
    )
    if not plot_incomplete_cis:
        curves = curves.dropna()
    curves["sig_diff"] = (curves.bin < curves.lower) | (curves.bin > curves.upper)

    fig = go.Figure()

    # Separate the data into two groups based on the sig_diff values
    group1 = curves[curves.sig_diff == 0]
    group2 = curves[curves.sig_diff == 1]

    # Create a trace for each group
    for df, color in zip([group1, group2], ["blue", "red"]):
        fig.add_trace(
            go.Scatter(
                x=df.bin,
                y=df.label_prop_positive,
                mode="markers",
                marker={
                    "color": color,
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": df.upper - df.label_prop_positive,
                    "arrayminus": df.label_prop_positive - df.lower,
                    "color": color,
                },
            )
        )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={
            "color": "Grey",
            "width": 2,
        },
    )

    # Add histogram of the scores
    min_score = data[score_col].min()
    max_score = data[score_col].max()
    counts, bin_edges = np.histogram(
        data[score_col],
        bins=num_bins,
        range=(min_score, max_score),
    )
    counts = counts / np.sum(counts)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            opacity=0.6,
            marker_color="lightblue",
            width=(max_score - min_score) / num_bins,
        )
    )

    fig.update_layout(showlegend=False, template="plotly_white")
    fig.update_xaxes(title_text="Average Score in Bin", range=x_lim)
    fig.update_yaxes(title_text="Average Label", range=x_lim)

    return fig


def plot_calibration_curve_by_segment(
    data: pd.DataFrame,
    group_var: str,
    score_col: str,
    label_col: str,
    num_bins: int = 20,
    n_cols: int = 4,
    sample_weight_col: str | None = None,
    binning_method: str = "equispaced",
) -> go.Figure:
    assert binning_method in ["equispaced", "equisized"]

    binning_fun = (
        utils.make_equispaced_bins
        if binning_method == "equispaced"
        else utils.make_equisized_bins
    )

    agg_df = data.groupby(group_var).apply(
        lambda x: _compute_calibration_curve(
            x,
            score_col=score_col,
            label_col=label_col,
            num_bins=num_bins,
            sample_weight_col=sample_weight_col,
            binning_method=binning_fun,
        )
    )

    if agg_df.shape[0] == 0:
        return go.Figure()
    curves = agg_df.reset_index().dropna()

    curves["error_minus"] = curves.label_prop_positive - curves.lower
    curves["error_plus"] = curves.upper - curves.label_prop_positive

    def _list_by_groupvar(
        df: pd.DataFrame, group_var: str, data_col: str
    ) -> list[list[float]]:
        return [
            list(df[df[group_var] == g][data_col].values)
            for g in df[group_var].unique()
        ]

    groups = curves[group_var].unique()
    x_values = _list_by_groupvar(curves, group_var, "bin")
    y_values = _list_by_groupvar(curves, group_var, "label_prop_positive")
    y_error_minus = _list_by_groupvar(curves, group_var, "error_minus")
    y_error = _list_by_groupvar(curves, group_var, "error_plus")

    # Calculate the number of rows needed
    num_rows = max(math.ceil(len(groups) / n_cols), 1)

    # Create subplots for each group
    if isinstance(groups, np.ndarray):
        groups = groups.tolist()
        groups = [str(group) for group in groups]

    fig = sp.make_subplots(rows=num_rows, cols=n_cols, subplot_titles=groups)

    for i in range(len(groups)):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Add a 45-degree line to the plot
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={
                "color": "Grey",
                "width": 2,
            },
            row=row,
            col=col,
        )

        # Check if the error bars overlap with the 45-degree line
        x_overlap = []
        y_overlap = []
        y_minus_overlap = []
        y_plus_overlap = []
        x_no_overlap = []
        y_no_overlap = []
        y_minus_no_overlap = []
        y_plus_no_overlap = []
        for j in range(len(y_values[i])):
            if (
                (y_values[i][j] - y_error_minus[i][j])
                <= x_values[i][j]
                <= (y_values[i][j] + y_error[i][j])
            ):
                x_overlap.append(x_values[i][j])
                y_overlap.append(y_values[i][j])
                y_minus_overlap.append(y_error_minus[i][j])
                y_plus_overlap.append(y_error[i][j])
            else:
                x_no_overlap.append(x_values[i][j])
                y_no_overlap.append(y_values[i][j])
                y_minus_no_overlap.append(y_error_minus[i][j])
                y_plus_no_overlap.append(y_error[i][j])

        # Add traces for points that overlap with the 45-degree line
        fig.add_trace(
            go.Scatter(
                x=x_overlap,
                y=y_overlap,
                mode="markers",  # Only display markers
                marker={"color": "blue"},  # Set marker color
                error_y={
                    "type": "data",
                    "array": y_plus_overlap,
                    "arrayminus": y_minus_overlap,
                    "visible": True,
                    "color": "blue",  # Set error bar color
                },
                name=f"{groups[i]} (overlap)",
            ),
            row=row,
            col=col,
        )

        # Add traces for points that do not overlap with the 45-degree line
        fig.add_trace(
            go.Scatter(
                x=x_no_overlap,
                y=y_no_overlap,
                mode="markers",  # Only display markers
                marker={"color": "red"},  # Set marker color
                error_y={
                    "type": "data",
                    "array": y_plus_no_overlap,
                    "arrayminus": y_minus_no_overlap,
                    "visible": True,
                    "color": "red",  # Set error bar color
                },
                name=f"{groups[i]} (no overlap)",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Average Score in Bin", range=[0, 1.1])
    fig.update_yaxes(title_text="Average Label", range=[0, 1.1])

    return fig


def plot_segment_calibration_errors(
    mce: metrics.MulticalibrationError,
    highlight_feature: str | None = None,
    quantity: str = "segment_ecces",
) -> go.Figure:
    """
    Plots a multi-segment Kuiper statistic scatter plot. This visualization helps in assessing the
    calibration error across different segments of data, which can be defined by categorical and
    numerical features.

    :param mce: A MulticalibrationError object
    :param highlight_feature: An optional feature to highlight in the plot.
    :param quantity: The quantity to plot. Options are "segment_ecces", "segment_p_values", "segment_sigmas",
        and "segment_ecces_sigma_scale"
    :returns: A Plotly Figure object representing the scatter plot of the Kuiper statistic
        against segment size, with optional highlighting of a specific feature.
    """
    valid_quantities = [
        "segment_ecces",
        "segment_ecces_absolute",
        "segment_p_values",
        "segment_sigmas",
        "segment_ecces_sigma_scale",
    ]
    if quantity not in valid_quantities:
        raise ValueError(
            f"Invalid quantity {quantity}. Options are {valid_quantities}."
        )
    segment_mask, segment_feature_values = mce.segments
    segment_mask = segment_mask.reshape(-1, segment_mask.shape[-1])
    categorical_segment_columns = mce.categorical_segment_columns or []
    numerical_segment_columns = mce.numerical_segment_columns or []
    all_eval_cols = categorical_segment_columns + numerical_segment_columns

    dicts = []
    mce_quantity = getattr(mce, quantity)
    for segment_idx in range(mce.total_number_segments):
        mask = segment_mask[segment_idx]
        value = mce_quantity[segment_idx]
        segment_features = segment_feature_values[
            segment_feature_values["idx_segment"] == segment_idx
        ].drop(columns=["idx_segment"])
        this_dict = {quantity: value, "segment_size": mask.sum()}
        for val_col in all_eval_cols:
            try:
                this_dict[val_col] = segment_features[
                    segment_features.segment_column == val_col
                ]["value"].values[0]
            except IndexError:
                this_dict[val_col] = "_all_"
        dicts.append(this_dict)

    plot_data = pd.DataFrame(dicts)

    fig_args = {
        "data_frame": plot_data,
        "x": "segment_size",
        "y": quantity,
        "hover_data": all_eval_cols,
        "log_x": True,
    }
    if highlight_feature is not None:
        fig_args["color"] = highlight_feature
    fig = px.scatter(**fig_args)
    fig.update_xaxes(title="Segment Size")

    if quantity == "segment_ecces":
        fig.update_yaxes(title="ECCE").update_layout(yaxis={"ticksuffix": "%"})
    elif quantity == "segment_sigmas":
        fig.update_yaxes(title="Standard deviation")
    elif quantity == "segment_p_values":
        fig.update_yaxes(title="P-value")
    elif quantity == "segment_ecces_sigma_scale":
        fig.update_yaxes(title="ECCE / Standard Deviation").update_layout(
            yaxis={"ticksuffix": "\u03c3"}
        )

    return fig


def plot_learning_curve(
    mcgrad_model: methods.MCGrad, show_all: bool = False
) -> go.Figure:
    """
    Plots a learning curve for an MCGrad model.

    :param mcgrad_model: An MCGrad model object.
    :param show_all: Whether to show all metrics in the learning curve. If False, only the metric specified in the model's early_stopping_score_func is shown.
    :returns: A Plotly Figure object representing the learning curve.
    """
    if not mcgrad_model.early_stopping:
        raise ValueError(
            "Learning curve can only be plotted for models that have been trained with early_stopping=True."
        )

    performance_metrics = mcgrad_model._performance_metrics
    extra_evaluation_due_to_early_stopping = (
        1
        if (
            mcgrad_model.early_stopping
            and len(mcgrad_model.mr) < mcgrad_model.num_rounds
        )
        else 0
    )
    # Calculate the total number of rounds (including the initial round)
    tot_num_rounds = min(
        1
        + len(mcgrad_model.mr)
        + extra_evaluation_due_to_early_stopping
        + mcgrad_model.patience,
        1 + mcgrad_model.num_rounds,
    )
    x_vals = np.arange(0, tot_num_rounds)
    metric_names = [mcgrad_model.early_stopping_score_func.name]
    for metric_name in performance_metrics.keys():
        if (
            "valid" in metric_name
            and mcgrad_model.early_stopping_score_func.name not in metric_name
            and show_all
        ):
            metric_names.append(metric_name.split("performance_")[-1])

    # Create subplots
    fig = make_subplots(
        rows=len(metric_names),
        cols=1,
        vertical_spacing=0.03,
        shared_xaxes=True,
    )

    colors = ["#007bff", "#dc3545"]  # Blue and Red

    for i, metric_name in enumerate(metric_names):
        test_performance = performance_metrics[f"avg_valid_performance_{metric_name}"]
        max_perf_for_annotation = np.max(test_performance)
        # Plot the test performance (held-out validation set)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=test_performance,
                mode="lines+markers",
                name="Held-out Validation Set" if i == 0 else "",
                line={"color": colors[0]},
                marker={
                    "color": colors[0],
                    "opacity": 1,
                    "symbol": "star",
                    "size": 12,
                },
                showlegend=False if i > 0 else True,
            ),
            row=i + 1,
            col=1,
        )
        if mcgrad_model.save_training_performance:
            train_performance = mcgrad_model._performance_metrics[
                f"avg_train_performance_{metric_name}"
            ]
            # Plot the training performance (training set)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=train_performance,
                    mode="lines+markers",
                    name="Training Set" if i == 0 else None,
                    line={"color": colors[1]},
                    marker={
                        "color": colors[1],
                        "opacity": 1,
                        "symbol": "star",
                        "size": 12,
                    },
                    showlegend=False if i > 0 else True,
                ),
                row=i + 1,
                col=1,
            )
            # Required for plotting the horizontal lines for MCE
            max_perf_for_annotation = max(
                max_perf_for_annotation, np.max(train_performance)
            )
        # Add vertical line for the selected iteration
        fig.add_vline(
            x=len(mcgrad_model.mr),
            line_dash="dash",
            line_color="black",
            opacity=0.5,
            row=i + 1,
            col=1,
        )
        if i == 0:
            # Add annotation for the selected iteration
            fig.add_annotation(
                text="Selected round by early stopping",
                xref="x",
                yref="y",
                x=len(mcgrad_model.mr) - 0.05,
                y=max_perf_for_annotation * 1.075,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font={"size": 12, "color": "black"},
                bordercolor="black",
                borderwidth=1,
                borderpad=1,
                bgcolor="lightgrey",
                opacity=0.7,
                row=i + 1,
                col=1,
            )
        if "mce_sigma_scale" in metric_name:
            if max_perf_for_annotation >= mcgrad_model.MCE_STRONG_EVIDENCE_THRESHOLD:
                fig.add_hline(
                    y=mcgrad_model.MCE_STRONG_EVIDENCE_THRESHOLD,
                    line_dash="dash",
                    line_color="darkgreen",
                    opacity=1,
                    row=i + 1,
                    col=1,
                    annotation_text="Strong<br>Miscalibration<br>.<br>.",
                    annotation_position="right",
                    annotation_font_color="darkgreen",
                    annotation_font_size=12,
                    annotation_textangle=90,
                )

            if max_perf_for_annotation >= mcgrad_model.MCE_STAT_SIGN_THRESHOLD:
                fig.add_hline(
                    y=mcgrad_model.MCE_STAT_SIGN_THRESHOLD,
                    line_dash="dash",
                    line_color="darkorange",
                    opacity=0.7,
                    row=i + 1,
                    col=1,
                    annotation_text="Stat. Significant<br>Miscalibration",
                    annotation_position="right",
                    annotation_font_color="darkorange",
                    annotation_font_size=12,
                    annotation_textangle=90,
                )

            if (
                test_performance[len(mcgrad_model.mr)]
                >= mcgrad_model.MCE_STRONG_EVIDENCE_THRESHOLD
            ):
                fig.add_annotation(
                    text=f"<b>WARNING: {mcgrad_model.__class__.__name__} run failed to remove strong evidence of multicalibration!</b>",
                    xref="paper",
                    yref="paper",
                    x=tot_num_rounds - 1,
                    y=min(test_performance)
                    + (max(test_performance) - min(test_performance)) / 2,
                    xanchor="right",
                    yanchor="middle",
                    showarrow=False,
                    font={"size": 12, "color": "darkred"},
                    bordercolor="darkred",
                    borderwidth=2,
                    borderpad=2,
                    bgcolor="yellow",
                    opacity=0.85,
                    row=i + 1,
                    col=1,
                )
    # Create the title and legend
    fig.update_layout(
        title_text="Learning Curves",
        font={"size": 16, "color": "#7f7f7f"},
        height=300 * len(metric_names),  # Adjust height based on number of metrics
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "left",
            "x": 0.0,
        },
    )

    # Update y-axis labels with metric names
    for i, metric_name in enumerate(metric_names):
        fig.update_yaxes(
            title_text=metric_name,
            row=i + 1,
            col=1,
        )
        # Update x-axis labels including "without MCGrad" for the 0th iteration
        if len(x_vals) <= 10:
            fig.update_xaxes(
                title_text=f"{mcgrad_model.__class__.__name__} round"
                if i == len(metric_names) - 1
                else "",
                tickmode="array",
                tickvals=x_vals,
                ticktext=[f"without<br>{mcgrad_model.__class__.__name__}"]
                + [str(int(val)) for val in x_vals[1:]],
                row=i + 1,
                col=1,
            )
        else:
            x_vals = np.arange(0, len(x_vals), np.ceil(len(x_vals) / 5))
            fig.update_xaxes(
                title_text=f"{mcgrad_model.__class__.__name__} round"
                if i == len(metric_names) - 1
                else "",
                tickmode="array",
                tickvals=x_vals,
                ticktext=[f"without<br>{mcgrad_model.__class__.__name__}"]
                + [str(int(val)) for val in x_vals[1:]],
                row=i + 1,
                col=1,
            )

    return fig
