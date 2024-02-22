import io
import os
from datetime import datetime

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image
from decode_mcd import DesignTargets, ContinuousTarget

from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer
from mcd_clip.optimization.embedding_similarity_optimizer import to_full_clips_dataframe
from mcd_clip.resource_utils import run_result_path
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


def render_some(full_df: pd.DataFrame, run_dir: str, batch_number: int):
    batch_dir = os.path.join(run_dir, f"batch_{batch_number}")
    os.makedirs(batch_dir, exist_ok=False)
    clips = to_full_clips_dataframe(CombinedDataset(full_df).get_as_clips())
    images_paths = []
    for idx in clips.index:
        rendering_result = IMAGE_CONVERTOR.to_image(clips.loc[idx])
        image_path = os.path.join(batch_dir, f"bike_{idx}.svg")
        images_paths.append(image_path)
        with open(image_path, "wb") as file:
            file.write(rendering_result.image)
    average_image(images_paths, batch_dir)


def get_validity(sampled: pd.DataFrame):
    result = pd.DataFrame(index=sampled.index)
    for i in range(len(COMBINED_VALIDATION_FUNCTIONS)):
        validation_function = COMBINED_VALIDATION_FUNCTIONS[i]
        validation_result = validation_function(sampled)
        result[f"validation_result_{i}"] = validation_result
    assert len(result) == len(sampled)
    return result


def run(features_off: bool):
    GENERATIONS = 50
    BATCH_SIZE = 50
    BATCHES = GENERATIONS // BATCH_SIZE

    run_id = str(datetime.now().strftime('%m-%d--%H.%M.%S')) + '-ablation-template'
    if features_off:
        run_id += '-features-off'

    optimizer = CombinedOptimizer(
        design_targets=DesignTargets(
            continuous_targets=[
                ContinuousTarget("Model Mass", 0, 7),
                ContinuousTarget("Sim 1 Safety Factor (Inverted)", 0, 1),
            ]
        ),
        target_embeddings=[],
        extra_bonus_objectives=['Model Mass', 'Sim 1 Safety Factor (Inverted)'],
    )
    optimizer.set_starting_design_by_index('1')
    features_desired = not features_off
    empty_repair_desired = features_off
    generator = optimizer.build_generator(validation_functions=COMBINED_VALIDATION_FUNCTIONS,
                                          gower_on=features_desired,
                                          average_gower_on=features_desired,
                                          changed_feature_on=features_desired,
                                          initialize_from_dataset=False,
                                          use_empty_repair=empty_repair_desired,
                                          )

    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)

    for i in range(1, BATCHES + 1):
        cumulative = i * BATCH_SIZE
        generator.generate(cumulative, seed=23)

        sampled = generator.sample_with_weights(num_samples=3,
                                                cfc_weight=100,
                                                gower_weight=100,
                                                avg_gower_weight=200,
                                                bonus_objectives_weights=np.array([1, 1]).reshape((1, 2)),
                                                diversity_weight=0.05,
                                                include_dataset=False)
        validity = get_validity(sampled)
        full_df = pd.concat([sampled, optimizer.predict(CombinedDataset(sampled)), validity], axis=1)
        assert len(full_df) == len(sampled)
        full_df.to_csv(os.path.join(run_dir, f"cfs_{i}.csv"))
        render_some(full_df, run_dir, i)


if __name__ == '__main__':
    run(features_off=True)
    run(features_off=False)
