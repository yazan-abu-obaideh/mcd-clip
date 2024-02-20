import io
import os
import random
from datetime import datetime

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image
from decode_mcd import DesignTargets, ContinuousTarget, CounterfactualsGenerator

from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, ARMPIT_WRIST_TARGET, KNEE_TARGET, \
    AERODYNAMIC_DRAG_TARGET
from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, distance_column_name, TextEmbeddingTarget, \
    ImageEmbeddingTarget, get_scores_dataframe
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.result_plots.draw_pair_plots import lyle_plot
from mcd_clip.singletons import IMAGE_CONVERTOR


def _generate_with_retry(cumulative, generator, seed=23):
    try:
        generator.generate(cumulative, seed=seed)
    except AssertionError as e:
        print(f"MCD failed while generating {e}, changing seed...")
        _generate_with_retry(cumulative, generator, seed=random.randint(1, 50))


def average_image(images_paths, batch_dir: str):
    # Create a numpy array of floats to store the average
    first_image = cairosvg.svg2png(url=images_paths[0])
    first_image = Image.open(io.BytesIO(first_image))
    w, h = first_image.size
    N = len(images_paths)
    arr = np.zeros((h, w, 3), ).astype('float64')

    # Convert SVG images to PNG using cairosvg and build up average pixel intensities
    for im in images_paths:
        png_image = cairosvg.svg2png(url=im)
        png_image = Image.open(io.BytesIO(png_image))
        imarr = np.array(png_image).astype('float64')
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save(os.path.join(batch_dir, "average.png"))


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
    average_image(images_paths, batch_dir)


def _build_full_df(generator: CounterfactualsGenerator,
                   optimizer: CombinedOptimizer,
                   starting_design: pd.DataFrame):
    sampled = generator.sample_with_weights(num_samples=1000,
                                            cfc_weight=1,
                                            gower_weight=1,
                                            avg_gower_weight=1,
                                            diversity_weight=0.1)
    with_query = pd.concat([sampled, starting_design], axis=0)
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


def run():
    GENERATIONS = 150
    BATCH_SIZE = 50
    BATCHES = GENERATIONS // BATCH_SIZE

    target_embeddings = [
        TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
        ImageEmbeddingTarget(image_path=resource_path('mtb.png'))
    ]
    design_targets = DesignTargets(
        continuous_targets=[
            ContinuousTarget('Sim 1 Safety Factor (Inverted)', lower_bound=0, upper_bound=0.76),
            ContinuousTarget('Model Mass', lower_bound=2, upper_bound=4),
            ContinuousTarget('ergonomic_score', lower_bound=0, upper_bound=47),
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
        full_df = _build_full_df(generator, optimizer, starting_design)
        full_df.to_csv(os.path.join(run_dir, f'batch_{i}.csv'))
        # lyle_plot(full_df,
        #           generator._problem._data_package.predictions_dataset.columns,
        #           generator._problem._data_package.design_targets.continuous_targets,
        #           os.path.join(run_dir, f"lyle_fig_batch_{i}.png"))
        render_some(full_df, run_dir, i)


if __name__ == '__main__':
    run()
