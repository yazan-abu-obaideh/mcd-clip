import os.path

import matplotlib.text
import pandas as pd
import seaborn as sns
from decode_mcd import DesignTargets, ContinuousTarget

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


def draw_figure():
    original_data = pd.read_csv(run_result_path(os.path.join('full-scores-fit-bikes-18807', 'batch_3_cfs.csv')),
                                index_col=0)
    selected_columns = ['Model Mass', 'Aerodynamic Drag', 'Sim 1 Safety Factor (Inverted)',
                        'gower_distance', 'avg_gower_distance', 'changed_feature_ratio']
    original_data = pd.DataFrame(original_data, columns=selected_columns)
    grid = sns.pairplot(original_data)
    for i in range(len(grid.axes)):
        for j in range(len(grid.axes[i])):
            _draw_on_axis(original_data, grid.axes[i][j])
    grid.savefig(run_result_path('fig-lines.png'))


def _draw_on_axis(original_data: pd.DataFrame, curr_axis):
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


if __name__ == '__main__':
    draw_figure()
