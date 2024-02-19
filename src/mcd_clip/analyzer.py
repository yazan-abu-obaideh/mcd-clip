import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
from decode_mcd import DesignTargets, ContinuousTarget

from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, ARMPIT_WRIST_TARGET, KNEE_TARGET, \
    AERODYNAMIC_DRAG_TARGET
from mcd_clip.datasets.combined_datasets import CombinedDataset, OriginalCombinedDataset
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, TextEmbeddingTarget, ImageEmbeddingTarget
from mcd_clip.resource_utils import run_result_path, resource_path
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


if __name__ == '__main__':
    get_worst()
    render_by_original_index('1276')
