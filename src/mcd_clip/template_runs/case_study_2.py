import io
import os
import random
from datetime import datetime
from typing import List

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image
from decode_mcd import DesignTargets, ContinuousTarget, CounterfactualsGenerator

from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, distance_column_name, TextEmbeddingTarget, \
    ImageEmbeddingTarget
from mcd_clip.optimization.embedding_similarity_optimizer import to_full_clips_dataframe
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.singletons import IMAGE_CONVERTOR


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


def render_some(full_df: pd.DataFrame, run_dir: str, batch_number: int, distance_column_suffix: str):
    batch_dir = os.path.join(run_dir, f"batch_{batch_number}_distance_{distance_column_suffix}")
    os.makedirs(batch_dir, exist_ok=False)
    print(f"Closest counterfactuals {full_df}")
    clips = to_full_clips_dataframe(CombinedDataset(full_df).get_as_clips())
    images_paths = []
    for idx in clips.index:
        rendering_result = IMAGE_CONVERTOR.to_image(clips.loc[idx])
        image_path = os.path.join(batch_dir, f"bike_{idx}.svg")
        images_paths.append(image_path)
        with open(image_path, "wb") as file:
            file.write(rendering_result.image)
    average_image(images_paths, batch_dir)


def run():
    TEXT_TARGET = "A futuristic black cyberpunk-style road racing bicycle"
    GENERATIONS = 400
    BATCH_SIZE = 100
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
                                                            in optimizer._starting_dataset.get_combined().columns
                                                            if 'bottle' not in f
                                                            ]
                                          )
    generator.use_empty_repair(False)

    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)

    for i in range(1, BATCHES + 1):
        cumulative = i * BATCH_SIZE
        _generate_with_retry(cumulative, generator)
        sampled = generator.sample_with_weights(num_samples=1000, cfc_weight=1, gower_weight=1, avg_gower_weight=1,
                                                diversity_weight=0.1)
        full_df = pd.concat([sampled, optimizer.predict(CombinedDataset(sampled))], axis=1)
        assert len(full_df) == len(sampled)
        full_df.to_csv(os.path.join(run_dir, f'batch_{i}.csv'))
        render_some(_sample(optimizer, generator, 0), run_dir, i, 'text')
        render_some(_sample(optimizer, generator, 1), run_dir, i, 'image')
        render_some(_balance_sample(generator,
                                    n_samples=3,
                                    objective_weights=[1, 1],
                                    diversity_weight=0.1),
                    run_dir, i, 'both')


def _sample(
        optimizer: CombinedOptimizer,
        generator: CounterfactualsGenerator, distance_column_index: int):
    as_many = _balance_sample(generator,
                              objective_weights=[1 * (1 - distance_column_index), 1 * distance_column_index],
                              diversity_weight=0.01,
                              n_samples=500)
    column_ = distance_column_name(distance_column_index)
    as_many[column_] = optimizer.predict(CombinedDataset(as_many))[column_]
    return as_many.sort_values(by=column_, ascending=True)[:3]


def _balance_sample(generator: CounterfactualsGenerator,
                    objective_weights: List[int],
                    diversity_weight: float,
                    n_samples: int):
    return generator.sample_with_weights(num_samples=n_samples, cfc_weight=1,
                                         gower_weight=1, avg_gower_weight=1,
                                         bonus_objectives_weights=np.array(objective_weights).reshape((1, 2)),
                                         diversity_weight=diversity_weight, include_dataset=False)


def _generate_with_retry(cumulative: int, generator: CounterfactualsGenerator, seed: int = 92):
    try:
        generator.generate(cumulative, seed=seed)
    except AssertionError as e:
        print(f"Error {e}, retrying...")
        _generate_with_retry(cumulative, generator, random.randint(1, 100))


if __name__ == '__main__':
    run()
