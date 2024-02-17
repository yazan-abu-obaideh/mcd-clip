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
        rendering_result = IMAGE_CONVERTOR.to_image(clips_data.loc[idx])
        with open(run_result_path(f'bike_{idx}.svg'), 'wb') as file:
            file.write(rendering_result.image)


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
        starting_design_index=0,
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


if __name__ == '__main__':
    get_predictions()
