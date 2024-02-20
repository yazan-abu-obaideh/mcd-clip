import io
import os.path

import cairosvg
import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
from PIL import Image
from decode_mcd import DesignTargets, ContinuousTarget
from matplotlib import pyplot as plt

from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, ARMPIT_WRIST_TARGET, KNEE_TARGET, \
    AERODYNAMIC_DRAG_TARGET
from mcd_clip.datasets.combined_datasets import CombinedDataset, OriginalCombinedDataset
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, TextEmbeddingTarget, ImageEmbeddingTarget, \
    distance_column_name
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.result_plots.draw_pair_plots import lyle_plot
from mcd_clip.singletons import IMAGE_CONVERTOR


def render_from_combined_data(data: pd.DataFrame):
    clips_data = CombinedDataset(data).get_as_clips()
    for idx in clips_data.index:
        _render_and_save(clips_data, idx)


def _render_and_save(clips_data: pd.DataFrame, idx):
    rendering_result = IMAGE_CONVERTOR.to_image(clips_data.loc[idx])
    with open(run_result_path(f'bike_{idx}.svg'), 'wb') as file:
        file.write(rendering_result.image)
    with open(run_result_path(f'bike_{idx}.txt'), 'w') as file:
        file.write(rendering_result.bike_xml)


def get_predictions():
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
            AERODYNAMIC_DRAG_TARGET
        ])
    combined_optimizer = CombinedOptimizer(
        extra_bonus_objectives=[],
        design_targets=design_targets,
        target_embeddings=target_embeddings
    )
    combined_dataset = OriginalCombinedDataset().get_combined_dataset()
    predictions = pd.DataFrame(combined_optimizer.predict(combined_dataset),
                               index=combined_dataset.get_combined().index)
    result = pd.concat([combined_dataset.get_combined(), predictions], axis=1)
    assert len(result) == len(combined_dataset.get_combined())
    result.to_csv('with_predictions.csv')


def get_worst():
    data = pd.read_csv('with_predictions.csv', index_col=0)
    data['bad_score'] = np.zeros(shape=(len(data),))
    for column in ['Model Mass', 'Sim 1 Safety Factor (Inverted)',
                   'embedding_distance_1', 'embedding_distance_2']:
        data['bad_score'] = data['bad_score'] + (data[column] / data[column].mean())
    ranked = data.sort_values(by='bad_score', ascending=False)
    ranked = ranked[ranked['Sim 1 Safety Factor (Inverted)'] < 1.5]
    ranked = ranked[ranked['Aerodynamic Drag'] != float('inf')]
    ranked = ranked[ranked['Model Mass'] < 10]

    for idx in ranked.index[5:25]:
        print(idx)
        element = ranked.loc[idx]
        print(f"{element['Model Mass']=}")
        print(f"{element['Sim 1 Safety Factor (Inverted)']=}")
        print(f"{element['embedding_distance_1']=}")
        print(f"{element['embedding_distance_2']=}")
        print(f"{element['Aerodynamic Drag']=}")
        print(f"{element['Knee Extension']=}")
        print(f"{element['Back Angle']=}")
        print(f"{element['Armpit Angle']=}")
        print(f"{element['Wheel width rear']=}")
        print(f"{element['SEATSTAYbrdgdia1']=}")
        print(f"{element['CHAINSTAYbrdgdia1']=}")

    return ranked


def render_by_original_index(original_index: str):
    clips_df = OriginalCombinedDataset().get_combined_dataset().get_as_clips()
    _render_and_save(clips_df, original_index)


def draw_lyle_plot():
    target_embeddings = [
        TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
        ImageEmbeddingTarget(image_path=resource_path('mtb.png'))
    ]

    full_df = pd.read_csv(
        run_result_path(os.path.join('02-20--06.54.16-template-combined-run', 'batch_3.csv')), index_col=0)

    full_df.rename(columns={'ergonomic_score': 'Ergonomic Factor (Inverted)'},
                   inplace=True)

    design_targets = DesignTargets(
        continuous_targets=[
            # ContinuousTarget('Sim 1 Safety Factor (Inverted)', lower_bound=0, upper_bound=1),
            # ContinuousTarget('Model Mass', lower_bound=2, upper_bound=5.5),
            # ContinuousTarget('Ergonomic Factor (Inverted)', lower_bound=0, upper_bound=60),
            # ContinuousTarget(label="Aerodynamic Drag", lower_bound=0, upper_bound=22.5),
            # ContinuousTarget(label=distance_column_name(0), lower_bound=0, upper_bound=0.74),
            # ContinuousTarget(label=distance_column_name(1), lower_bound=0, upper_bound=0.15),
            ContinuousTarget('gower_distance', lower_bound=0, upper_bound=1),
            ContinuousTarget('avg_gower_distance', lower_bound=0, upper_bound=1),
            ContinuousTarget('changed_feature_ratio', lower_bound=0, upper_bound=1),
        ])

    mcd_scores = ['gower_distance', 'avg_gower_distance', 'changed_feature_ratio']
    columns = [
        'Sim 1 Safety Factor (Inverted)', 'Model Mass', 'embedding_distance_1', 'embedding_distance_2',
        'Aerodynamic Drag',
        'Ergonomic Factor (Inverted)',
        'Knee Extension', 'Back Angle', 'Armpit Angle',
    ]
    lyle_plot(
        full_df,
        prediction_columns=columns + mcd_scores,
        continuous_targets=design_targets.continuous_targets,
        save_path=run_result_path('lyle-plot-scores.png')
    )


def draw_bikes_grid():
    run_dir = run_result_path('02-20--06.27.53-template-A futuristic black cyberpunk-style road racing bicycle (copy)')
    batch = '3'
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
    draw_lyle_plot()
