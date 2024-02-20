import os.path
from typing import List

import matplotlib.text
import pandas as pd
import seaborn as sns
from decode_mcd import DesignTargets, ContinuousTarget, CounterfactualsGenerator

import matplotlib.patches as patches
from matplotlib.path import Path

from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, ARMPIT_WRIST_TARGET, KNEE_TARGET
from mcd_clip.optimization.combined_optimizer import TextEmbeddingTarget, ImageEmbeddingTarget
from mcd_clip.resource_utils import run_result_path, resource_path

target_embeddings = [
    TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
    ImageEmbeddingTarget(image_path=resource_path('mtb.png'))
]
design_targets = DesignTargets(
    continuous_targets=[
        ContinuousTarget('Sim 1 Safety Factor (Inverted)', lower_bound=0, upper_bound=1),
        ContinuousTarget('Model Mass', lower_bound=0, upper_bound=6),
        BACK_TARGET,
        ARMPIT_WRIST_TARGET,
        KNEE_TARGET,
        # AERODYNAMIC_DRAG_TARGET
    ])


def _get_x_label(_axis) -> matplotlib.text.Text:
    return _axis.xaxis.label


def _get_y_label(_axis) -> matplotlib.text.Text:
    return _axis.yaxis.label


def draw_figure(data: pd.DataFrame, selected_columns: List[str], save_path: str):
    trimmed_data = pd.DataFrame(data, columns=selected_columns)
    grid = sns.pairplot(trimmed_data)
    for i in range(len(grid.axes)):
        for j in range(len(grid.axes[i])):
            _draw_on_axis(grid.axes[i][j])
    grid.savefig(save_path)


def _draw_on_axis(curr_axis):
    x_label = _get_x_label(curr_axis).get_text()
    y_label = _get_y_label(curr_axis).get_text()
    if x_label in design_targets.get_all_constrained_labels():
        print(f"Drawing line for {x_label} and {y_label}")
        x_target = [target for
                    target in design_targets.continuous_targets
                    if target.label == x_label
                    ][0]
        x_target: ContinuousTarget
        curr_axis.axvline(
            x=x_target.lower_bound,
        )
        curr_axis.axvline(
            x=x_target.upper_bound,
        )


def regular_plot():
    pass


def lyle_plot(counterfactuals: pd.DataFrame,
              prediction_columns: List[str],
              continuous_targets: List[DesignTargets],
              save_path: str):
    # replace with get_predictions
    column_names = prediction_columns
    obj_scores = pd.DataFrame(counterfactuals, columns=column_names)
    scores = []
    minimums = []
    maximums = []
    for i, target in enumerate(continuous_targets):
        name = target.label
        score = obj_scores[name]
        scores.append(score)
        minimums.append(target.lower_bound)
        maximums.append(target.upper_bound)
    scores = pd.concat(scores, axis=1)
    # print(scores)
    num = len(scores.columns)
    # add column to distinguish query
    classes = ["Counterfactual"] * (len(counterfactuals) - 1) + ["Query"]
    scores['class'] = classes
    # seaborn pairplot of validity
    palette = ["#3291a8", "#000000"]
    markers = [".", "X"]
    s = 20
    grid = sns.pairplot(scores, hue="class", kind="scatter", diag_kind="kde", palette=palette, markers=markers,
                        plot_kws={"s": s}, diag_kws={"cut": 0})
    proxy = [patches.Patch(color='gray', alpha=0.5, label='Shaded Region')]

    # add shaded region to plot
    for i, ax in enumerate(grid.axes.ravel()):
        x_idx = i % num
        y_idx = i // num
        minx_i = max(minimums[x_idx], ax.get_xlim()[0])
        maxx_i = min(maximums[x_idx], ax.get_xlim()[1])
        miny_i = max(minimums[y_idx], ax.get_ylim()[0])
        maxy_i = min(maximums[y_idx], ax.get_ylim()[1])
        minx_o = ax.get_xlim()[0]
        maxx_o = ax.get_xlim()[1]
        miny_o = ax.get_ylim()[0]
        maxy_o = ax.get_ylim()[1]

        if x_idx != y_idx:
            vertices = [
                (minx_o, miny_o),  # Botton left of outer rect
                (minx_o, maxy_o),  # Top left
                (maxx_o, maxy_o),  # Top right
                (maxx_o, miny_o),  # Bottom right
                (minx_o, miny_o),  # Bottom left again to close the path
                (minx_i, miny_i),  # Botton left of inner rect
                (maxx_i, miny_i),  # Bottom right
                (maxx_i, maxy_i),  # Top right
                (minx_i, maxy_i),  # Top left
                (minx_i, miny_i),  # Bottom left again to close the path
            ]

            codes = [
                Path.MOVETO,  # Start path
                Path.LINETO,  # Line to top-left of outer rectangle
                Path.LINETO,  # Line to top-right
                Path.LINETO,  # Line to bottom-right
                Path.CLOSEPOLY,  # Close the outer rectangle
                Path.MOVETO,  # Start path for the inner rectangle
                Path.LINETO,  # Line to bottom-right of inner rectangle
                Path.LINETO,  # Line to top-right
                Path.LINETO,  # Line to top-left
                Path.CLOSEPOLY,  # Close the inner rectangle (creates the cut-out effect)
            ]

            # Combine vertices and codes into a path object
            path = Path(vertices, codes)

            # Create a patch based on the path object
            patch = patches.PathPatch(path, facecolor='gray', lw=0, alpha=0.2)
            ax.add_patch(patch)
            pass
        else:
            rect = patches.Rectangle((minx_i, miny_o), maxx_i - minx_i, maxy_o - miny_o, linewidth=0, edgecolor='black',
                                     facecolor='gray', alpha=0.2)
            ax.add_patch(rect)
            pass

    # add shaded region to legend
    legend = grid.fig.legends[0]  # Get the first (and typically only) legend
    handles, labels = legend.legendHandles, [text.get_text() for text in legend.texts]
    handles.append(patch)
    labels.append('Invalid Region')

    # Remove the existing legend
    for legend in grid.fig.legends:
        legend.remove()
    # Create replacement
    grid.fig.legend(handles, labels, loc='upper right', title='Legend')

    grid.savefig(save_path)
