import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from decode_mcd import DesignTargets, ContinuousTarget, CounterfactualsGenerator

from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, ARMPIT_WRIST_TARGET, KNEE_TARGET, \
    AERODYNAMIC_DRAG_TARGET
from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, distance_column_name, TextEmbeddingTarget, \
    ImageEmbeddingTarget, get_scores_dataframe
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.result_plots.draw_pair_plots import custom_plot
from mcd_clip.singletons import IMAGE_CONVERTOR


def _generate_with_retry(cumulative, generator, seed=23):
    try:
        generator.generate(cumulative, seed=seed)
    except AssertionError as e:
        print(f"MCD failed while generating {e}, changing seed...")
        _generate_with_retry(cumulative, generator, seed=random.randint(1, 50))


def render_some(full_df: pd.DataFrame, run_dir: str, batch_number: int):
    batch_dir = os.path.join(run_dir, f"batch_{batch_number}")
    os.makedirs(batch_dir, exist_ok=False)
    sampled_counterfactuals = full_df.sort_values(by=distance_column_name(0), ascending=True)[:5]
    print(f"Closest counterfactuals {sampled_counterfactuals}")
    clips = CombinedDataset(sampled_counterfactuals).get_as_clips()
    images_paths = []
    for idx in clips.index:
        rendering_result = IMAGE_CONVERTOR.to_image(clips.loc[idx])
        image_path = os.path.join(batch_dir, f"bike_{idx}.svg")
        images_paths.append(image_path)
        with open(image_path, "wb") as file:
            file.write(rendering_result.image)


def _build_cfs_with_query(generator: CounterfactualsGenerator,
                          optimizer: CombinedOptimizer,
                          starting_design: pd.DataFrame):
    sampled = generator.sample_with_weights(num_samples=1000,
                                            cfc_weight=1,
                                            gower_weight=1,
                                            avg_gower_weight=1,
                                            diversity_weight=0.1)
    with_query = pd.concat([starting_design, sampled, ], axis=0)
    scores_df = get_scores_dataframe(generator, with_query)
    scores_df.index = with_query.index
    predictions = optimizer.predict(CombinedDataset(with_query))
    predictions.index = with_query.index
    full_df = pd.concat(
        [with_query,
         predictions,
         scores_df
         ],
        axis=1
    )
    assert len(full_df) == len(sampled) + 1
    return full_df


def build_dataset_with_predictions(combined_optimizer: CombinedOptimizer):
    dataset = combined_optimizer.starting_dataset
    df = dataset.get_combined()
    predictions = combined_optimizer.predict(dataset)
    for c in predictions.columns:
        df[c] = predictions[c]
    df.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df


def run():
    GENERATIONS = 30
    BATCH_SIZE = 30
    BATCHES = GENERATIONS // BATCH_SIZE

    target_embeddings = [
        TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
        ImageEmbeddingTarget(image_path=resource_path('mtb.png'))
    ]
    design_targets = DesignTargets(
        continuous_targets=[
            ContinuousTarget('Sim 1 Safety Factor (Inverted)', lower_bound=0, upper_bound=1.5),
            ContinuousTarget('Model Mass', lower_bound=2, upper_bound=8),
            ContinuousTarget('ergonomic_score', lower_bound=0, upper_bound=75),
            BACK_TARGET,
            ARMPIT_WRIST_TARGET,
            KNEE_TARGET,
            ContinuousTarget(label="Aerodynamic Drag", lower_bound=0, upper_bound=22.5),
            ContinuousTarget(label=distance_column_name(0), lower_bound=0, upper_bound=0.73),
            ContinuousTarget(label=distance_column_name(1), lower_bound=0, upper_bound=0.11),
        ]

    )

    extra_bonus_objective = ["Model Mass", "ergonomic_score", AERODYNAMIC_DRAG_TARGET.label]

    run_id = str(datetime.now().strftime('%m-%d--%H.%M.%S')) + "-template-" + 'combined-run'

    optimizer = CombinedOptimizer(
        design_targets=design_targets,
        target_embeddings=target_embeddings,
        extra_bonus_objectives=extra_bonus_objective
    )
    optimizer.set_starting_design_by_index('1548')
    generator = optimizer.build_generator(validation_functions=COMBINED_VALIDATION_FUNCTIONS)

    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)

    starting_design = optimizer.starting_design.get_combined()
    starting_design.index = ['query']

    for i in range(1, BATCHES + 1):
        cumulative = i * BATCH_SIZE
        _generate_with_retry(cumulative, generator)
        full_df = _build_cfs_with_query(generator, optimizer, starting_design)
        full_df.to_csv(os.path.join(run_dir, f'batch_{i}.csv'))
        custom_plot(full_df,
                    build_dataset_with_predictions(optimizer).sample(300),
                    generator._problem._data_package.predictions_dataset.columns,
                    generator._problem._data_package.design_targets.continuous_targets,
                    os.path.join(run_dir, os.path.join(run_dir, f"lyle_fig_batch_{i}.png")))
        render_some(full_df, run_dir, i)


if __name__ == '__main__':
    run()
