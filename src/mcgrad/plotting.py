# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

"""
Visualization tools for calibration analysis.

This module provides functions for creating diagnostic plots to assess and
visualize the calibration quality of probabilistic predictions. All plots
are generated using Plotly for interactive visualization.
"""

import math
from typing import Any, get_args, Literal, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _utils as utils, methods, metrics
from ._compat import groupby_apply
from ._utils import BinningMethodInterface

BinningMethod = Literal["equispaced", "equisized"]

_MARKER_COLOR_CALIBRATED = "blue"
_MARKER_COLOR_MISCALIBRATED = "red"


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


def _get_binning_function(
    binning_method: BinningMethod,
) -> BinningMethodInterface:
    if binning_method == "equispaced":
        return utils.make_equispaced_bins
    elif binning_method == "equisized":
        return utils.make_equisized_bins
    else:
        raise ValueError(
            f"Invalid binning_method '{binning_method}'. "
            "Must be 'equispaced' or 'equisized'."
        )


def plot_global_calibration_curve(
    data: pd.DataFrame,
    score_col: str,
    label_col: str,
    num_bins: int = metrics.CALIBRATION_ERROR_NUM_BINS,
    sample_weight_col: str | None = None,
    binning_method: BinningMethod = "equispaced",
    plot_incomplete_cis: bool = True,
    x_lim: Tuple[float, float] = (0, 1.1),
) -> go.Figure:
    """
    Plots a global calibration curve with confidence intervals and score histogram.

    The calibration curve shows the relationship between predicted scores and actual
    label proportions. Calibrated bins (where the diagonal falls within the confidence
    interval) are shown in blue, while miscalibrated bins are shown in red.

    :param data: DataFrame containing scores and labels.
    :param score_col: Column name for the predicted scores.
    :param label_col: Column name for the binary labels.
    :param num_bins: Number of bins for the calibration curve.
    :param sample_weight_col: Optional column name for sample weights.
    :param binning_method: Method for binning scores. Either "equispaced" (equal-width
        bins) or "equisized" (equal-count bins).
    :param plot_incomplete_cis: Whether to plot bins with incomplete confidence
        intervals (i.e., bins with NaN values).
    :param x_lim: Tuple specifying the x-axis and y-axis limits.
    :return: A Plotly Figure object with the calibration curve, confidence intervals,
        and a histogram of scores.
    """
    binning_fun = _get_binning_function(binning_method)

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
    curves["is_miscalibrated"] = (curves.bin < curves.lower) | (
        curves.bin > curves.upper
    )

    fig = go.Figure()

    calibrated_bins = curves[~curves.is_miscalibrated]
    miscalibrated_bins = curves[curves.is_miscalibrated]

    for df, color in [
        (calibrated_bins, _MARKER_COLOR_CALIBRATED),
        (miscalibrated_bins, _MARKER_COLOR_MISCALIBRATED),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.bin,
                y=df.label_prop_positive,
                mode="markers",
                marker={"color": color},
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
        line={"color": "Grey", "width": 2},
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
    binning_method: BinningMethod = "equispaced",
) -> go.Figure:
    """
    Plots calibration curves for each segment defined by a grouping variable.

    Creates a grid of subplots, one per unique value of the grouping variable.
    Each subplot shows a calibration curve with confidence intervals. Calibrated
    bins are shown in blue, miscalibrated bins in red.

    :param data: DataFrame containing scores, labels, and the grouping variable.
    :param group_var: Column name for the grouping variable that defines segments.
    :param score_col: Column name for the predicted scores.
    :param label_col: Column name for the binary labels.
    :param num_bins: Number of bins for each calibration curve.
    :param n_cols: Number of columns in the subplot grid.
    :param sample_weight_col: Optional column name for sample weights.
    :param binning_method: Method for binning scores. Either "equispaced" (equal-width
        bins) or "equisized" (equal-count bins).
    :return: A Plotly Figure object with a grid of calibration curve subplots.
    """
    binning_fun = _get_binning_function(binning_method)

    agg_df = groupby_apply(
        data.groupby(group_var),
        lambda x: _compute_calibration_curve(
            x,
            score_col=score_col,
            label_col=label_col,
            num_bins=num_bins,
            sample_weight_col=sample_weight_col,
            binning_method=binning_fun,
        ),
    )

    if agg_df.shape[0] == 0:
        return go.Figure()
    curves = agg_df.reset_index().dropna()

    curves["error_minus"] = curves.label_prop_positive - curves.lower
    curves["error_plus"] = curves.upper - curves.label_prop_positive

    groups = list(curves[group_var].unique())

    num_rows = max(math.ceil(len(groups) / n_cols), 1)
    fig = make_subplots(
        rows=num_rows, cols=n_cols, subplot_titles=[str(g) for g in groups]
    )

    for i, group in enumerate(groups):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"color": "Grey", "width": 2},
            row=row,
            col=col,
        )

        group_data = curves[curves[group_var] == group]
        is_calibrated = (
            (group_data.label_prop_positive - group_data.error_minus) <= group_data.bin
        ) & (group_data.bin <= (group_data.label_prop_positive + group_data.error_plus))

        for mask, color, suffix in [
            (is_calibrated, _MARKER_COLOR_CALIBRATED, "calibrated"),
            (~is_calibrated, _MARKER_COLOR_MISCALIBRATED, "miscalibrated"),
        ]:
            subset = group_data[mask]
            fig.add_trace(
                go.Scatter(
                    x=subset.bin.tolist(),
                    y=subset.label_prop_positive.tolist(),
                    mode="markers",
                    marker={"color": color},
                    error_y={
                        "type": "data",
                        "array": subset.error_plus.tolist(),
                        "arrayminus": subset.error_minus.tolist(),
                        "visible": True,
                        "color": color,
                    },
                    name=f"{group} ({suffix})",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Average Score in Bin", range=[0, 1.1])
    fig.update_yaxes(title_text="Average Label", range=[0, 1.1])

    return fig


SegmentQuantity = Literal[
    "segments_ecce_relative",
    "segments_ecce",
    "segments_ecce_pvalue",
    "segments_ecce_sigma",
]
_VALID_SEGMENT_QUANTITIES: tuple[str, ...] = get_args(SegmentQuantity)


def plot_segment_calibration_errors(
    mce: metrics.MulticalibrationError,
    highlight_feature: str | None = None,
    quantity: SegmentQuantity = "segments_ecce_relative",
) -> go.Figure:
    """
    Plots a segment-level calibration error scatter plot.

    This visualization displays the specified calibration quantity against segment size,
    helping to assess calibration across different data segments defined by categorical
    and numerical features.

    :param mce: A MulticalibrationError object containing computed segment-level metrics.
    :param highlight_feature: Optional feature name to color-code points by.
    :param quantity: The quantity to plot. Options:

        - ``segments_ecce_relative``: ECCE as percentage of prevalence (default)
        - ``segments_ecce``: Absolute ECCE values
        - ``segments_ecce_pvalue``: P-values for calibration test
        - ``segments_ecce_sigma``: ECCE in standard deviations

    :return: A Plotly Figure object with the scatter plot of the specified quantity
        against segment size.
    """
    if quantity not in _VALID_SEGMENT_QUANTITIES:
        raise ValueError(
            f"Invalid quantity '{quantity}'. Options are {_VALID_SEGMENT_QUANTITIES}."
        )
    segment_mask, segment_feature_values = mce._segments
    segment_mask = segment_mask.reshape(-1, segment_mask.shape[-1])
    categorical_segment_columns = mce.categorical_segment_columns or []
    numerical_segment_columns = mce.numerical_segment_columns or []
    all_eval_cols = categorical_segment_columns + numerical_segment_columns

    mce_quantity = getattr(mce, quantity)
    rows = []
    for segment_idx in range(mce.total_number_segments):
        mask = segment_mask[segment_idx]
        segment_features = segment_feature_values[
            segment_feature_values["idx_segment"] == segment_idx
        ].drop(columns=["idx_segment"])

        row = {quantity: mce_quantity[segment_idx], "segment_size": mask.sum()}
        for col in all_eval_cols:
            matching = segment_features[segment_features.segment_column == col]
            row[col] = matching["value"].values[0] if len(matching) > 0 else "_all_"
        rows.append(row)

    plot_data = pd.DataFrame(rows)

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

    y_axis_config = {
        "segments_ecce_relative": ("ECCE (relative)", "%"),
        "segments_ecce": ("ECCE (absolute)", None),
        "segments_ecce_pvalue": ("P-value", None),
        "segments_ecce_sigma": ("ECCE (sigma)", "\u03c3"),
    }
    title, suffix = y_axis_config[quantity]
    fig.update_yaxes(title=title)
    if suffix:
        fig.update_layout(yaxis={"ticksuffix": suffix})

    return fig


_LEARNING_CURVE_COLORS = {"validation": "#007bff", "training": "#dc3545"}


def plot_learning_curve(
    mcgrad_model: methods.MCGrad, show_all: bool = False
) -> go.Figure:
    """
    Plots a learning curve for an MCGrad model.

    :param mcgrad_model: An MCGrad model object.
    :param show_all: Whether to show all metrics in the learning curve.
        If False, only the metric specified in the model's early_stopping_score_func
        is shown.
    :returns: A Plotly Figure object representing the learning curve.
    """
    if not mcgrad_model.early_stopping:
        raise ValueError(
            "Learning curve can only be plotted for models trained with "
            "early_stopping=True."
        )

    performance_metrics = mcgrad_model._performance_metrics
    stopped_early = len(mcgrad_model.mr) < mcgrad_model.num_rounds
    extra_eval = 1 if stopped_early else 0

    tot_num_rounds = min(
        1 + len(mcgrad_model.mr) + extra_eval + mcgrad_model.patience,
        1 + mcgrad_model.num_rounds,
    )
    x_vals = np.arange(0, tot_num_rounds)

    metric_names = [mcgrad_model.early_stopping_score_func.name]
    if show_all:
        for metric_name in performance_metrics:
            if (
                "valid" in metric_name
                and mcgrad_model.early_stopping_score_func.name not in metric_name
            ):
                metric_names.append(metric_name.split("performance_")[-1])

    fig = make_subplots(
        rows=len(metric_names),
        cols=1,
        vertical_spacing=0.03,
        shared_xaxes=True,
    )

    selected_round = len(mcgrad_model.mr)
    model_name = mcgrad_model.__class__.__name__

    for i, metric_name in enumerate(metric_names):
        row_num = i + 1
        is_first_row = i == 0
        is_last_row = i == len(metric_names) - 1

        valid_perf = performance_metrics[f"avg_valid_performance_{metric_name}"]
        max_perf = np.max(valid_perf)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=valid_perf,
                mode="lines+markers",
                name="Held-out Validation Set" if is_first_row else "",
                line={"color": _LEARNING_CURVE_COLORS["validation"]},
                marker={
                    "color": _LEARNING_CURVE_COLORS["validation"],
                    "symbol": "star",
                    "size": 12,
                },
                showlegend=is_first_row,
            ),
            row=row_num,
            col=1,
        )

        if mcgrad_model.save_training_performance:
            train_perf = performance_metrics[f"avg_train_performance_{metric_name}"]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=train_perf,
                    mode="lines+markers",
                    name="Training Set" if is_first_row else None,
                    line={"color": _LEARNING_CURVE_COLORS["training"]},
                    marker={
                        "color": _LEARNING_CURVE_COLORS["training"],
                        "symbol": "star",
                        "size": 12,
                    },
                    showlegend=is_first_row,
                ),
                row=row_num,
                col=1,
            )
            max_perf = max(max_perf, np.max(train_perf))

        fig.add_vline(
            x=selected_round,
            line_dash="dash",
            line_color="black",
            opacity=0.5,
            row=row_num,
            col=1,
        )

        if is_first_row:
            fig.add_annotation(
                text="Selected round by early stopping",
                xref="x",
                yref="y",
                x=selected_round - 0.05,
                y=max_perf * 1.075,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font={"size": 12, "color": "black"},
                bordercolor="black",
                borderwidth=1,
                borderpad=1,
                bgcolor="lightgrey",
                opacity=0.7,
                row=row_num,
                col=1,
            )

        if "mce_sigma" in metric_name:
            _add_mce_threshold_annotations(
                fig, mcgrad_model, valid_perf, max_perf, row_num, tot_num_rounds
            )

        fig.update_yaxes(title_text=metric_name, row=row_num, col=1)
        _update_x_axis_ticks(
            fig, x_vals.tolist(), model_name, row_num, is_last_row=is_last_row
        )

    fig.update_layout(
        title_text="Learning Curves",
        font={"size": 16, "color": "#7f7f7f"},
        height=300 * len(metric_names),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "left",
            "x": 0.0,
        },
    )

    return fig


def _add_mce_threshold_annotations(
    fig: go.Figure,
    model: methods.MCGrad,
    valid_perf: list[Any],
    max_perf: float,
    row: int,
    tot_rounds: int,
) -> None:
    if max_perf >= model.MCE_STRONG_EVIDENCE_THRESHOLD:
        fig.add_hline(
            y=model.MCE_STRONG_EVIDENCE_THRESHOLD,
            line_dash="dash",
            line_color="darkgreen",
            opacity=1,
            row=row,
            col=1,
            annotation_text="Strong<br>Miscalibration<br>.<br>.",
            annotation_position="right",
            annotation_font_color="darkgreen",
            annotation_font_size=12,
            annotation_textangle=90,
        )

    if max_perf >= model.MCE_STAT_SIGN_THRESHOLD:
        fig.add_hline(
            y=model.MCE_STAT_SIGN_THRESHOLD,
            line_dash="dash",
            line_color="darkorange",
            opacity=0.7,
            row=row,
            col=1,
            annotation_text="Stat. Significant<br>Miscalibration",
            annotation_position="right",
            annotation_font_color="darkorange",
            annotation_font_size=12,
            annotation_textangle=90,
        )

    selected_round = len(model.mr)
    if valid_perf[selected_round] >= model.MCE_STRONG_EVIDENCE_THRESHOLD:
        fig.add_annotation(
            text=f"<b>WARNING: {model.__class__.__name__} run failed to remove "
            "strong evidence of multicalibration!</b>",
            xref="paper",
            yref="paper",
            x=tot_rounds - 1,
            y=min(valid_perf) + (max(valid_perf) - min(valid_perf)) / 2,
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            font={"size": 12, "color": "darkred"},
            bordercolor="darkred",
            borderwidth=2,
            borderpad=2,
            bgcolor="yellow",
            opacity=0.85,
            row=row,
            col=1,
        )


def _update_x_axis_ticks(
    fig: go.Figure,
    x_vals: list[Any],
    model_name: str,
    row: int,
    is_last_row: bool,
) -> None:
    title = f"{model_name} round" if is_last_row else ""

    if len(x_vals) > 10:
        x_vals = list(np.arange(0, len(x_vals), int(np.ceil(len(x_vals) / 5))))

    fig.update_xaxes(
        title_text=title,
        tickmode="array",
        tickvals=x_vals,
        ticktext=[f"without<br>{model_name}"] + [str(int(v)) for v in x_vals[1:]],
        row=row,
        col=1,
    )
