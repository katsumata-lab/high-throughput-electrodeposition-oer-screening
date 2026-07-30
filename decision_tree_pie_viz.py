from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PALETTE = ("#8fd14f", "#ffbf00", "#1da1e6", "#ef476f", "#9b5de5", "#06d6a0")

__all__ = [
    "DecisionTreePieVizConfig",
    "plot_decision_tree_pies",
    "save_decision_tree_pies",
]

MAX_RENDER_PIXELS_PER_SIDE = 32000


@dataclass(slots=True)
class DecisionTreePieVizConfig:
    pie_bins: int = 4
    binning: str = "quantile"
    palette: tuple[str, ...] = DEFAULT_PALETTE
    figure_facecolor: str = "#141414"
    transparent_background: bool = False
    text_color: str = "#101010"
    font_scale: float = 1.0
    min_node_radius_px: float = 18.0
    max_node_radius_px: float = 56.0
    donut_hole_ratio: float = 0.42
    edge_base_width: float = 1.8
    edge_max_width: float = 9.8
    figure_width_per_leaf: float = 2.2
    figure_height_per_depth: float = 2.8
    min_figure_width: float = 12.0
    min_figure_height: float = 8.0
    node_spacing_padding_px: float = 36.0


def _format_interval(interval: object) -> str:
    if hasattr(interval, "left") and hasattr(interval, "right"):
        return f"{interval.left:.1f} to {interval.right:.1f}"
    return str(interval)


def _display_comparator(comparator: str) -> str:
    if comparator == "<=":
        return "≦"
    return comparator


def _make_target_bins(y, num_bins: int, method: str):
    import pandas as pd

    unique_count = y.nunique(dropna=True)
    num_bins = max(1, min(num_bins, unique_count))

    if num_bins == 1:
        labels = ["all samples"]
        values = np.zeros(len(y), dtype=int)
        return values, labels

    if unique_count <= num_bins:
        categories = pd.Categorical(y.astype(str))
        labels = categories.categories.tolist()
        values = categories.codes
        return values, labels

    if method == "uniform":
        binned = pd.cut(y, bins=num_bins, include_lowest=True, duplicates="drop")
    else:
        binned = pd.qcut(y, q=num_bins, duplicates="drop")

    categories = list(binned.cat.categories)
    labels = [_format_interval(cat) for cat in categories]
    values = binned.cat.codes.to_numpy()
    return values, labels


def _build_node_samples(tree_, X_array: np.ndarray):
    node_to_indices: dict[int, np.ndarray] = {}

    def recurse(node_id: int, indices: np.ndarray) -> None:
        node_to_indices[node_id] = indices

        left_id = tree_.children_left[node_id]
        right_id = tree_.children_right[node_id]
        if left_id == right_id:
            return

        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        left_mask = X_array[indices, feature_idx] <= threshold
        recurse(left_id, indices[left_mask])
        recurse(right_id, indices[~left_mask])

    recurse(0, np.arange(len(X_array)))
    return node_to_indices


def _compute_layout(tree_):
    positions: dict[int, tuple[float, float, int]] = {}
    next_leaf_x = 0
    max_depth = 0

    def recurse(node_id: int, depth: int) -> float:
        nonlocal next_leaf_x, max_depth
        max_depth = max(max_depth, depth)
        left_id = tree_.children_left[node_id]
        right_id = tree_.children_right[node_id]

        if left_id == right_id:
            x = float(next_leaf_x)
            next_leaf_x += 1
        else:
            left_x = recurse(left_id, depth + 1)
            right_x = recurse(right_id, depth + 1)
            x = (left_x + right_x) / 2.0

        positions[node_id] = (x, -float(depth), depth)
        return x

    recurse(0, 0)

    leaf_count = max(1, next_leaf_x)
    normalized = {}
    for node_id, (x, y, depth) in positions.items():
        normalized[node_id] = ((x + 0.5) / leaf_count, y, depth)
    return normalized, max_depth, leaf_count


def _ellipsize(text: str, width: int = 22) -> str:
    wrapped_lines = []
    for line in text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=False) or [""])
    return "\n".join(wrapped_lines)


def _data_to_axes_fraction(ax, center_data):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_frac = (center_data[0] - x_min) / (x_max - x_min)
    y_frac = (center_data[1] - y_min) / (y_max - y_min)
    return x_frac, y_frac


def _scaled_font_size(config: DecisionTreePieVizConfig, size: float) -> float:
    return size * config.font_scale


def _make_label_box(*, alpha: float = 0.3, pad: float = 0.28) -> dict[str, float | str]:
    return {
        "boxstyle": f"round,pad={pad},rounding_size=0.05",
        "fc": "white",
        "ec": "none",
        "alpha": alpha,
    }


def _pixels_to_axes_delta(ax, dx_px: float = 0.0, dy_px: float = 0.0) -> tuple[float, float]:
    axes_bbox = ax.get_window_extent()
    width_px = max(float(axes_bbox.width), 1.0)
    height_px = max(float(axes_bbox.height), 1.0)
    return dx_px / width_px, dy_px / height_px


def _nudge_axes_text(ax, text_artist, dx_px: float = 0.0, dy_px: float = 0.0) -> None:
    x_pos, y_pos = text_artist.get_position()
    dx_axes, dy_axes = _pixels_to_axes_delta(ax, dx_px=dx_px, dy_px=dy_px)
    text_artist.set_position((x_pos + dx_axes, y_pos + dy_axes))


def _set_axes_text_position(text_artist, x_pos: float | None = None, y_pos: float | None = None) -> None:
    current_x, current_y = text_artist.get_position()
    text_artist.set_position(
        (
            current_x if x_pos is None else x_pos,
            current_y if y_pos is None else y_pos,
        )
    )


def _bbox_overlaps(bbox_a, bbox_b, pad_px: float = 4.0) -> bool:
    return not (
        bbox_a.x1 + pad_px < bbox_b.x0
        or bbox_b.x1 + pad_px < bbox_a.x0
        or bbox_a.y1 + pad_px < bbox_b.y0
        or bbox_b.y1 + pad_px < bbox_a.y0
    )


def _separate_axes_texts(
    fig,
    ax,
    text_a,
    text_b,
    *,
    move_a_px: tuple[float, float],
    move_b_px: tuple[float, float],
    pad_px: float = 4.0,
    max_iterations: int = 24,
) -> None:
    for _ in range(max_iterations):
        renderer = fig.canvas.get_renderer()
        bbox_a = text_a.get_window_extent(renderer=renderer)
        bbox_b = text_b.get_window_extent(renderer=renderer)
        if not _bbox_overlaps(bbox_a, bbox_b, pad_px=pad_px):
            break
        _nudge_axes_text(ax, text_a, dx_px=move_a_px[0], dy_px=move_a_px[1])
        _nudge_axes_text(ax, text_b, dx_px=move_b_px[0], dy_px=move_b_px[1])


def _resolve_text_overlaps(
    fig,
    ax,
    positions,
    branch_label_artists,
    node_label_artists,
    sample_label_artists,
    leaf_label_artists,
) -> None:
    for node_id, branch_entries in branch_label_artists.items():
        sorted_entries = sorted(branch_entries, key=lambda entry: entry[0])
        if len(sorted_entries) == 2:
            left_branch = sorted_entries[0][1]
            right_branch = sorted_entries[1][1]
            _separate_axes_texts(
                fig,
                ax,
                left_branch,
                right_branch,
                move_a_px=(-10.0, 4.0),
                move_b_px=(10.0, 4.0),
                pad_px=6.0,
            )

        sample_label = sample_label_artists.get(node_id)
        node_label = node_label_artists.get(node_id)
        if sample_label is not None and node_label is not None:
            _separate_axes_texts(
                fig,
                ax,
                sample_label,
                node_label,
                move_a_px=(0.0, 8.0),
                move_b_px=(0.0, -10.0),
                pad_px=6.0,
            )

        for direction, branch_label in sorted_entries:
            branch_dx = 12.0 * direction
            if sample_label is not None:
                _separate_axes_texts(
                    fig,
                    ax,
                    branch_label,
                    sample_label,
                    move_a_px=(branch_dx, 5.0),
                    move_b_px=(0.0, 0.0),
                    pad_px=6.0,
                )
            if node_label is not None:
                _separate_axes_texts(
                    fig,
                    ax,
                    branch_label,
                    node_label,
                    move_a_px=(branch_dx, 6.0),
                    move_b_px=(0.0, -6.0),
                    pad_px=6.0,
                )

        if len(sorted_entries) == 2:
            left_branch = sorted_entries[0][1]
            right_branch = sorted_entries[1][1]
            aligned_y = max(left_branch.get_position()[1], right_branch.get_position()[1])
            _set_axes_text_position(left_branch, y_pos=aligned_y)
            _set_axes_text_position(right_branch, y_pos=aligned_y)

    for node_id, leaf_label in leaf_label_artists.items():
        sample_label = sample_label_artists.get(node_id)
        if sample_label is None:
            continue
        _separate_axes_texts(
            fig,
            ax,
            sample_label,
            leaf_label,
            move_a_px=(0.0, 6.0),
            move_b_px=(0.0, -10.0),
            pad_px=6.0,
        )

    depth_groups = {}
    for node_id, node_label in node_label_artists.items():
        depth = positions[node_id][2]
        depth_groups.setdefault(depth, []).append((positions[node_id][0], node_label))
    for entries in depth_groups.values():
        sorted_entries = sorted(entries, key=lambda entry: entry[0])
        for (left_x, left_label), (right_x, right_label) in zip(sorted_entries, sorted_entries[1:]):
            push_px = 8.0 if right_x >= left_x else -8.0
            _separate_axes_texts(
                fig,
                ax,
                left_label,
                right_label,
                move_a_px=(-abs(push_px), 0.0),
                move_b_px=(abs(push_px), 0.0),
                pad_px=6.0,
            )

    fig.canvas.draw()


def _iter_split_nodes(tree_):
    for node_id in range(tree_.node_count):
        left_id = tree_.children_left[node_id]
        right_id = tree_.children_right[node_id]
        if left_id != right_id:
            yield node_id, left_id, right_id


def _edge_linewidth(child_count: int, root_samples: int, config: DecisionTreePieVizConfig) -> float:
    return config.edge_base_width + (config.edge_max_width - config.edge_base_width) * (
        child_count / root_samples
    )


def _dominant_edge_color(node_indices, bin_codes, bin_label_count: int, pie_colors):
    child_bins = np.bincount(
        bin_codes[node_indices],
        minlength=bin_label_count,
    )
    return pie_colors[int(np.argmax(child_bins))]


def _draw_pie_patches(
    ax,
    center_axes,
    values,
    colors,
    radius_px,
    hole_ratio: float,
):
    from matplotlib.patches import Circle, Wedge
    from matplotlib.transforms import Affine2D

    total = int(np.sum(values))
    if total <= 0:
        return

    # Convert the requested radius from display pixels into axes-fraction units.
    # `ax.get_position()` returns figure-relative coordinates, not pixel sizes,
    # so using it here makes large radii explode far outside the canvas.
    axes_bbox = ax.get_window_extent()
    axes_width_px = max(float(axes_bbox.width), 1.0)
    axes_height_px = max(float(axes_bbox.height), 1.0)
    width_to_height = axes_width_px / axes_height_px
    radius_axes_x = radius_px / axes_width_px

    x_frac, y_frac = center_axes
    y_scaled = y_frac / width_to_height
    transform = Affine2D().scale(1.0, width_to_height) + ax.transAxes

    start_angle = 90.0
    for count, color in zip(values, colors):
        if count <= 0:
            continue

        end_angle = start_angle - 360.0 * (count / total)
        ax.add_patch(
            Wedge(
                center=(x_frac, y_scaled),
                r=radius_axes_x,
                theta1=end_angle,
                theta2=start_angle,
                width=radius_axes_x * (1.0 - hole_ratio),
                facecolor=color,
                edgecolor="white",
                linewidth=2.0,
                transform=transform,
                zorder=8,
            )
        )
        start_angle = end_angle

    ax.add_patch(
        Circle(
            (x_frac, y_scaled),
            radius=radius_axes_x,
            facecolor="none",
            edgecolor="white",
            linewidth=2.0,
            transform=transform,
            zorder=9,
        )
    )
    ax.add_patch(
        Circle(
            (x_frac, y_scaled),
            radius=radius_axes_x * hole_ratio,
            facecolor=ax.get_facecolor(),
            edgecolor="white",
            linewidth=2.0,
            transform=transform,
            zorder=9,
        )
    )


def _compute_node_scale_factor(centers_px, radii_px, min_gap_pixels: float) -> float:
    if len(centers_px) < 2:
        return 1.0

    max_scale = 1.0
    for i in range(len(centers_px)):
        x1, y1 = centers_px[i]
        r1 = radii_px[i]
        for j in range(i + 1, len(centers_px)):
            x2, y2 = centers_px[j]
            r2 = radii_px[j]
            distance = float(np.hypot(x2 - x1, y2 - y1))
            required = r1 + r2 + min_gap_pixels
            if required <= 0:
                continue
            pair_scale = distance / required
            if pair_scale < max_scale:
                max_scale = pair_scale

    return max(0.01, min(1.0, max_scale))


def _compute_node_radii_px(node_sample_counts, config: DecisionTreePieVizConfig):
    min_samples = min(node_sample_counts.values())
    max_samples = max(node_sample_counts.values())

    if min_samples == max_samples:
        midpoint = (config.min_node_radius_px + config.max_node_radius_px) / 2.0
        return {node_id: midpoint for node_id in node_sample_counts}

    node_radii_px = {}
    for node_id, sample_count in node_sample_counts.items():
        normalized = (sample_count - min_samples) / (max_samples - min_samples)
        node_radii_px[node_id] = config.min_node_radius_px + (
            config.max_node_radius_px - config.min_node_radius_px
        ) * normalized
    return node_radii_px


def _draw_split_edges_and_labels(
    ax,
    tree_,
    positions,
    node_to_indices,
    X_columns,
    bin_codes,
    bin_labels,
    pie_colors,
    root_samples: int,
    node_radii_px,
    config: DecisionTreePieVizConfig,
):
    label_box = _make_label_box(alpha=0.3, pad=0.28)
    node_label_box = _make_label_box(alpha=0.3, pad=0.35)
    branch_label_artists = {}
    node_label_artists = {}

    for node_id, left_id, right_id in _iter_split_nodes(tree_):
        parent_x, parent_y, _ = positions[node_id]
        parent_axes = _data_to_axes_fraction(ax, (parent_x, parent_y))
        feature_name = X_columns[tree_.feature[node_id]]
        threshold = tree_.threshold[node_id]
        node_radius_px = node_radii_px[node_id]

        label_offset_px = node_radius_px + 30.0 + 16.0 * config.font_scale
        _, label_offset_axes = _pixels_to_axes_delta(ax, dy_px=label_offset_px)
        node_label_artists[node_id] = ax.text(
            parent_axes[0],
            parent_axes[1] - label_offset_axes,
            _ellipsize(f"{feature_name}\n{threshold:.2f}", width=24),
            ha="center",
            va="top",
            fontsize=_scaled_font_size(config, 14),
            fontweight="bold",
            color="#101010",
            transform=ax.transAxes,
            bbox=node_label_box,
            zorder=6,
        )

        for child_id, comparator, direction in (
            (left_id, "<=", -1),
            (right_id, ">", 1),
        ):
            child_x, child_y, _ = positions[child_id]
            child_indices = node_to_indices[child_id]
            child_count = len(child_indices)
            edge_color = _dominant_edge_color(child_indices, bin_codes, len(bin_labels), pie_colors)

            ax.plot(
                [parent_x, child_x],
                [parent_y - 0.12, child_y + 0.18],
                color=edge_color,
                linewidth=_edge_linewidth(child_count, root_samples, config),
                alpha=0.95,
                solid_capstyle="round",
                zorder=1,
            )

            child_axes = _data_to_axes_fraction(ax, (child_x, child_y))
            mid_x = (parent_axes[0] + child_axes[0]) / 2.0
            mid_y = (parent_axes[1] + child_axes[1]) / 2.0
            branch_dx_axes, branch_dy_axes = _pixels_to_axes_delta(
                ax,
                dx_px=direction * (18.0 + 8.0 * config.font_scale),
                dy_px=12.0 + 8.0 * config.font_scale,
            )
            branch_artist = ax.text(
                mid_x + branch_dx_axes,
                mid_y + branch_dy_axes,
                f"{_display_comparator(comparator)} {threshold:.2f}",
                color="#101010",
                fontsize=_scaled_font_size(config, 12),
                fontweight="bold",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=label_box,
                zorder=5,
            )
            branch_label_artists.setdefault(node_id, []).append((direction, branch_artist))

    return branch_label_artists, node_label_artists


def _draw_target_legend(ax, bin_labels, pie_colors, target_name: str, config: DecisionTreePieVizConfig) -> None:
    legend_x = 0.02
    legend_y = 0.98
    ax.text(
        legend_x,
        legend_y + 0.04,
        f"Target bins for {target_name}",
        transform=ax.transAxes,
        color=config.text_color,
        fontsize=_scaled_font_size(config, 15),
        fontweight="bold",
        ha="left",
    )
    for idx, (label, color) in enumerate(zip(bin_labels, pie_colors)):
        y_pos = legend_y - idx * 0.045
        ax.scatter([legend_x], [y_pos], s=180, color=color, transform=ax.transAxes, clip_on=False, zorder=10)
        ax.text(
            legend_x + 0.03,
            y_pos,
            label,
            transform=ax.transAxes,
            color=config.text_color,
            fontsize=_scaled_font_size(config, 13),
            ha="left",
            va="center",
        )


def _draw_node_pies_and_leaf_labels(
    ax,
    tree_,
    ordered_node_ids,
    positions,
    node_to_indices,
    bin_codes,
    bin_labels,
    pie_colors,
    node_radii_px,
    config: DecisionTreePieVizConfig,
):
    label_box = _make_label_box(alpha=0.3, pad=0.28)
    sample_label_artists = {}
    leaf_label_artists = {}
    axes_height_px = ax.get_window_extent().height

    for node_id in ordered_node_ids:
        x_pos, y_pos, _ = positions[node_id]
        node_indices = node_to_indices[node_id]
        counts = np.bincount(bin_codes[node_indices], minlength=len(bin_labels))
        center_axes = _data_to_axes_fraction(ax, (x_pos, y_pos))

        _draw_pie_patches(
            ax,
            center_axes,
            counts,
            pie_colors,
            radius_px=node_radii_px[node_id],
            hole_ratio=config.donut_hole_ratio,
        )

        label_offset = (node_radii_px[node_id] + 10.0 + 6.0 * config.font_scale) / axes_height_px
        sample_label_artists[node_id] = ax.text(
            center_axes[0],
            center_axes[1] - label_offset,
            str(len(node_indices)),
            color="#101010",
            fontsize=_scaled_font_size(config, 12),
            fontweight="bold",
            ha="center",
            va="top",
            transform=ax.transAxes,
            bbox=label_box,
            zorder=10,
        )

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
        if is_leaf:
            prediction = float(tree_.value[node_id][0][0])
            leaf_label_offset = label_offset + (28.0 + 8.0 * config.font_scale) / axes_height_px
            leaf_label_artists[node_id] = ax.text(
                center_axes[0],
                center_axes[1] - leaf_label_offset,
                f"leaf\npred {prediction:.1f}",
                ha="center",
                va="top",
                fontsize=_scaled_font_size(config, 12),
                fontweight="bold",
                color="#101010",
                transform=ax.transAxes,
                bbox=label_box,
                zorder=6,
            )

    return sample_label_artists, leaf_label_artists


def plot_decision_tree_pies(
    model,
    X,
    y,
    *,
    target_name: str = "target",
    config: DecisionTreePieVizConfig | None = None,
    fig=None,
    ax=None,
):
    import matplotlib.pyplot as plt
    import pandas as pd

    config = config or DecisionTreePieVizConfig()
    if not hasattr(model, "tree_"):
        raise TypeError("model must be a fitted scikit-learn decision tree model.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame so feature names can be used.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    tree_ = model.tree_
    X_array = X.to_numpy(dtype=float)
    node_to_indices = _build_node_samples(tree_, X_array)
    positions, max_depth, leaf_count = _compute_layout(tree_)
    bin_codes, bin_labels = _make_target_bins(y, config.pie_bins, config.binning)
    pie_colors = list(config.palette[: len(bin_labels)])
    root_samples = len(X)
    node_sample_counts = {
        node_id: len(node_to_indices[node_id])
        for node_id in range(tree_.node_count)
    }
    node_radii_px = _compute_node_radii_px(node_sample_counts, config)

    if fig is None or ax is None:
        fig_width = max(config.min_figure_width, leaf_count * config.figure_width_per_leaf)
        fig_height = max(config.min_figure_height, (max_depth + 1) * config.figure_height_per_depth)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=config.figure_facecolor)
    if config.transparent_background:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.figure_facecolor)
    ax.set_xlim(0, 1)
    ax.set_ylim(-max_depth - 0.85, 0.35)
    ax.axis("off")
    fig.tight_layout()
    ordered_node_ids = list(range(tree_.node_count))
    fig.tight_layout()
    fig.canvas.draw()

    node_centers_axes = []
    node_centers_px = []
    node_radius_values_px = []
    for node_id in ordered_node_ids:
        x_pos, y_pos, _ = positions[node_id]
        center_axes = _data_to_axes_fraction(ax, (x_pos, y_pos))
        node_centers_axes.append(center_axes)
        node_centers_px.append(ax.transAxes.transform(center_axes))
        node_radius_values_px.append(node_radii_px[node_id])

    overlap_scale = _compute_node_scale_factor(
        node_centers_px,
        node_radius_values_px,
        min_gap_pixels=config.node_spacing_padding_px,
    )
    if overlap_scale < 1.0:
        resize_factor = 1.0 / overlap_scale
        current_width_px, current_height_px = fig.canvas.get_width_height()
        resized_width_px = current_width_px * resize_factor
        resized_height_px = current_height_px * resize_factor
        if (
            resized_width_px > MAX_RENDER_PIXELS_PER_SIDE
            or resized_height_px > MAX_RENDER_PIXELS_PER_SIDE
        ):
            raise ValueError(
                "Requested node radii are too large for this tree layout. "
                f"Estimated canvas size would be about {resized_width_px:.0f} x "
                f"{resized_height_px:.0f} px; reduce min/max node radius."
            )
        fig.set_size_inches(
            fig.get_figwidth() * resize_factor,
            fig.get_figheight() * resize_factor,
            forward=True,
        )
        fig.tight_layout()
        fig.canvas.draw()

    branch_label_artists, node_label_artists = _draw_split_edges_and_labels(
        ax,
        tree_,
        positions,
        node_to_indices,
        X.columns,
        bin_codes,
        bin_labels,
        pie_colors,
        root_samples,
        node_radii_px,
        config,
    )
    _draw_target_legend(ax, bin_labels, pie_colors, target_name, config)
    sample_label_artists, leaf_label_artists = _draw_node_pies_and_leaf_labels(
        ax,
        tree_,
        ordered_node_ids,
        positions,
        node_to_indices,
        bin_codes,
        bin_labels,
        pie_colors,
        node_radii_px,
        config,
    )

    fig.canvas.draw()
    _resolve_text_overlaps(
        fig,
        ax,
        positions,
        branch_label_artists,
        node_label_artists,
        sample_label_artists,
        leaf_label_artists,
    )

    return fig, ax


def save_decision_tree_pies(
    model,
    X,
    y,
    output_path: str | Path,
    *,
    target_name: str = "target",
    config: DecisionTreePieVizConfig | None = None,
    dpi: int = 300,
):
    config = config or DecisionTreePieVizConfig()
    fig, ax = plot_decision_tree_pies(model, X, y, target_name=target_name, config=config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        transparent=config.transparent_background,
    )
    return fig, ax, output_path
