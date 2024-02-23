import io
import os.path

import cairosvg
import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
from PIL import Image
from decode_mcd import DesignTargets, ContinuousTarget
from matplotlib import pyplot as plt

from mcd_clip.all_runs.ablation_runs import get_validity
from mcd_clip.datasets.combined_datasets import CombinedDataset, OriginalCombinedDataset
from mcd_clip.optimization.combined_optimizer import distance_column_name
from mcd_clip.optimization.embedding_similarity_optimizer import to_full_clips_dataframe
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.result_plots.draw_pair_plots import custom_plot
from mcd_clip.singletons import IMAGE_CONVERTOR


def render_from_combined_data(data: pd.DataFrame):
    clips_data = to_full_clips_dataframe(CombinedDataset(data).get_as_clips())
    for idx in clips_data.index:
        _render_and_save(clips_data, idx)


def print_validity():
    df = pd.read_csv(resource_path('all_structural_data_aug.csv'))
    validity = get_validity(df)
    print(validity.head())
    print(f"Fraction valid: {len(validity[np.sum(validity, axis=1) == 0])/len(validity)}")
    print(f"Average CV: {np.mean(np.sum(validity, axis=1))}")


def _render_and_save(clips_data: pd.DataFrame, idx):
    rendering_result = IMAGE_CONVERTOR.to_image(clips_data.loc[idx])
    with open(run_result_path(f'bike_{idx}.svg'), 'wb') as file:
        file.write(rendering_result.image)
    with open(run_result_path(f'bike_{idx}.txt'), 'w') as file:
        file.write(rendering_result.bike_xml)


def render_by_original_index(original_index: str):
    clips_df = OriginalCombinedDataset().get_combined_dataset().get_as_clips()
    _render_and_save(clips_df, original_index)


def draw_lyle_plot_thing():
    mcd_scores = ['gower_distance', 'avg_gower_distance', 'changed_feature_ratio']
    columns = [
        'Sim 1 Safety Factor (Inverted)', 'Model Mass', 'embedding_distance_1', 'embedding_distance_2',
        'Aerodynamic Drag',
        'ergonomic_score',
        'Knee Extension', 'Back Angle', 'Armpit Angle',
    ]

    design_targets = DesignTargets(
        continuous_targets=[
            ContinuousTarget(r'Safety Factor $\uparrow$', lower_bound=1.316, upper_bound=20),
            ContinuousTarget(r'Frame Mass $\downarrow$', lower_bound=2, upper_bound=4),
            ContinuousTarget(r'Ergonomics $\downarrow$', lower_bound=0, upper_bound=47),
            ContinuousTarget(label=r"Drag Force $\downarrow$", lower_bound=0, upper_bound=22.5),
            ContinuousTarget(label=r'Text Match $\downarrow$', lower_bound=0, upper_bound=0.73),
            ContinuousTarget(label=r'Image Match $\downarrow$', lower_bound=0, upper_bound=0.11),
            # ContinuousTarget('gower_distance', lower_bound=0, upper_bound=1),
            # ContinuousTarget('avg_gower_distance', lower_bound=0, upper_bound=1),
            # ContinuousTarget('changed_feature_ratio', lower_bound=0, upper_bound=1),
        ])

    counterfactuals_with_scores = pd.read_csv(
        run_result_path(os.path.join('02-21--01.09.38-template-combined-run', 'batch_3.csv')), index_col=0)

    old_to_new = {
        'Sim 1 Safety Factor (Inverted)': r'Safety Factor $\uparrow$',
        'Model Mass': r'Frame Mass $\downarrow$',
        'Aerodynamic Drag': r'Drag Force $\downarrow$',
        'ergonomic_score': r'Ergonomics $\downarrow$',
        distance_column_name(0): r'Text Match $\downarrow$',
        distance_column_name(1): r'Image Match $\downarrow$',
    }
    counterfactuals_with_scores.rename(columns=old_to_new, inplace=True)

    counterfactuals_with_scores[r'Safety Factor $\uparrow$'] = 1 / counterfactuals_with_scores[
        r'Safety Factor $\uparrow$']

    counterfactuals_with_scores = counterfactuals_with_scores[::-1]

    dataset = pd.read_csv(run_result_path('dataset_with_predictions.csv'), index_col=0)
    dataset.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
    dataset.dropna(axis=0, inplace=True)
    dataset.rename(columns=old_to_new, inplace=True)
    dataset[r'Safety Factor $\uparrow$'] = 1 / dataset[r'Safety Factor $\uparrow$']
    custom_plot(
        query_and_counterfactuals=counterfactuals_with_scores[:100],
        dataset_w_predictions=dataset.sample(300),
        continuous_targets=design_targets.continuous_targets,
        prediction_columns=list(old_to_new.values()),
        save_path='lyle-plot-latest.png'
    )


def draw_bikes_grid():
    run_dir = run_result_path('02-19--22.08.27-template-A yellow road-racing bicycle (copy)')
    batch = '4'
    images_grid = [[], [], []]
    goals = ['image', 'both', 'text']
    for i in range(3):
        goals_dir = os.path.join(run_dir, f'batch_{batch}_distance_{goals[i]}')
        image_paths = [im for im in os.listdir(goals_dir) if '.svg' in im]
        print(f"{image_paths=}")
        for j in range(3):
            joined = os.path.join(goals_dir, image_paths[2 - j])
            with open(joined, 'rb') as file:
                images_grid[i].append(file.read())

    n = 3

    fig, axs = plt.subplots(n, n, figsize=(15, 10))

    for i in range(n):
        for j in range(n):
            print("Rendering image...")
            ax = axs[i, j]
            image_open = Image.open(io.BytesIO(cairosvg.svg2png(images_grid[j][i])))
            ax.imshow(np.asarray(image_open, dtype='int32'))
            ax.axis('off')
    plt.savefig(run_result_path('bikes_array.png'))


if __name__ == '__main__':
    print_validity()
