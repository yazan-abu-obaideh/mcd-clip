import io
import os
from datetime import datetime

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image
from decode_mcd import DesignTargets, ContinuousTarget

from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, KNEE_TARGET, ARMPIT_WRIST_TARGET
from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, distance_column_name, TextEmbeddingTarget
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


def get_validity(sampled: pd.DataFrame):
    result = pd.DataFrame(index=sampled.index)
    for i in range(len(COMBINED_VALIDATION_FUNCTIONS)):
        validation_function = COMBINED_VALIDATION_FUNCTIONS[i]
        validation_result = validation_function(sampled)
        result[f"validation_result_{i}"] = validation_result
    assert len(result) == len(sampled)
    return result


def run(ablation: bool):
    TEXT_TARGET = "A pink road bike with water bottles"
    GENERATIONS = 250
    BATCH_SIZE = 50
    BATCHES = GENERATIONS // BATCH_SIZE

    run_id = str(datetime.now().strftime('%m-%d--%H.%M.%S')) + TEXT_TARGET + "-template-"
    if ablation:
        run_id += 'ablation'

    optimizer = CombinedOptimizer(
        design_targets=DesignTargets(
            continuous_targets=[
                ContinuousTarget(distance_column_name(0), 0, 1),
                BACK_TARGET,
                KNEE_TARGET,
                ARMPIT_WRIST_TARGET
            ]
        ),
        target_embeddings=[TextEmbeddingTarget(text_target=TEXT_TARGET)],
        extra_bonus_objectives=[],
    )
    optimizer.set_starting_design_by_index('1')
    features_desired = not ablation
    empty_repair_desired = ablation
    generator = optimizer.build_generator(validation_functions=[
    ],
        gower_on=features_desired,
        average_gower_on=features_desired,
        changed_feature_on=features_desired,
        use_empty_repair=empty_repair_desired
    )

    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)

    for i in range(1, BATCHES + 1):
        cumulative = i * BATCH_SIZE
        generator.generate(cumulative, seed=23)

        score_weight = 10

        sampled = generator.sample_with_weights(num_samples=100,
                                                cfc_weight=score_weight,
                                                gower_weight=score_weight,
                                                avg_gower_weight=score_weight,
                                                diversity_weight=0.1)
        validity = get_validity(sampled)
        full_df = pd.concat([sampled, optimizer.predict(CombinedDataset(sampled)), validity], axis=1)
        assert len(full_df) == len(sampled)
        full_df.to_csv(os.path.join(run_dir, f"cfs_{i}.csv"))
        render_some(full_df, run_dir, i)


if __name__ == '__main__':
    run(ablation=False)
    run(ablation=True)
