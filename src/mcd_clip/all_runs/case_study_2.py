import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from decode_mcd import DesignTargets, ContinuousTarget, CounterfactualsGenerator

from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, distance_column_name, TextEmbeddingTarget, \
    ImageEmbeddingTarget
from mcd_clip.optimization.embedding_similarity_optimizer import to_full_clips_dataframe
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.singletons import IMAGE_CONVERTOR


def render_some(full_df: pd.DataFrame, run_dir: str, batch_number: int, distance_column_suffix: str):
    batch_dir = os.path.join(run_dir, f"batch_{batch_number}_distance_{distance_column_suffix}")
    os.makedirs(batch_dir, exist_ok=False)
    clips = to_full_clips_dataframe(CombinedDataset(full_df).get_as_clips())
    images_paths = []
    for idx in clips.index:
        rendering_result = IMAGE_CONVERTOR.to_image(clips.loc[idx])
        image_path = os.path.join(batch_dir, f"bike_{idx}.svg")
        images_paths.append(image_path)
        with open(image_path, "wb") as file:
            file.write(rendering_result.image)


def run():
    TEXT_TARGET = "A futuristic black cyberpunk-style road racing bicycle"
    GENERATIONS = 800
    BATCH_SIZE = 200
    BATCHES = GENERATIONS // BATCH_SIZE

    run_id = str(datetime.now().strftime('%m-%d--%H.%M.%S')) + "-template-" + TEXT_TARGET

    optimizer = CombinedOptimizer(
        design_targets=DesignTargets(
            continuous_targets=[
                ContinuousTarget(distance_column_name(0), 0, 1),
                ContinuousTarget(distance_column_name(1), 0, 1),
            ]
        ),
        target_embeddings=[TextEmbeddingTarget(text_target=TEXT_TARGET),
                           ImageEmbeddingTarget(image_path=resource_path('mtb.png'))
                           ],
        extra_bonus_objectives=[]
    )
    optimizer.set_starting_design_by_index('1')
    generator = optimizer.build_generator(validation_functions=COMBINED_VALIDATION_FUNCTIONS,
                                          features_to_vary=[f for f
                                                            in optimizer.starting_dataset.get_combined().columns
                                                            if ('bottle' not in f)
                                                            ]
                                          )
    generator.use_empty_repair(False)

    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)

    for i in range(1, BATCHES + 1):
        cumulative = i * BATCH_SIZE
        generator.generate(cumulative, seed=45)
        render_some(_selective_sample(optimizer, generator, 0), run_dir, i, 'text')
        render_some(_selective_sample(optimizer, generator, 1), run_dir, i, 'image')
        render_some(_balance_sample(generator,
                                    n_samples=4,
                                    objective_weights=[1, 1],
                                    diversity_weight=0.1),
                    run_dir, i, 'both')


def _selective_sample(
        optimizer: CombinedOptimizer,
        generator: CounterfactualsGenerator,
        distance_column_index: int):
    desired_weight = 1
    objective_weights = [desired_weight * (1 - distance_column_index), desired_weight * distance_column_index]
    as_many = generator.sample_with_weights(num_samples=100, cfc_weight=0,
                                            gower_weight=0, avg_gower_weight=0,
                                            diversity_weight=0,
                                            bonus_objectives_weights=np.array(objective_weights).reshape((1, 2)),
                                            include_dataset=False)
    column_ = distance_column_name(distance_column_index)
    as_many[column_] = optimizer.predict(CombinedDataset(as_many))[column_]
    closes_cfs = as_many.sort_values(by=column_, ascending=True)[:4]
    print(f"Found closest cfs {closes_cfs[distance_column_name(distance_column_index)]}")
    return closes_cfs


def _balance_sample(generator: CounterfactualsGenerator,
                    objective_weights: List[int],
                    diversity_weight: float,
                    n_samples: int):
    return generator.sample_with_weights(num_samples=n_samples, cfc_weight=1,
                                         gower_weight=1, avg_gower_weight=1,
                                         bonus_objectives_weights=np.array(objective_weights).reshape((1, 2)),
                                         diversity_weight=diversity_weight, include_dataset=False)


if __name__ == '__main__':
    run()
