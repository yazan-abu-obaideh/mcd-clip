import os
import time
import uuid
from traceback import print_exception

import numpy as np
import pandas as pd
from decode_mcd import CounterfactualsGenerator

from mcd_clip.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from mcd_clip.bike_embedding.embedding_comparator import get_cosine_similarity
from mcd_clip.bike_embedding.embedding_similarity_optimizer import build_generator, to_full_dataframe, PREDICTOR
from mcd_clip.bike_rendering.parametric_to_image_convertor import ParametricToImageConvertor, RenderingResult
from mcd_clip.resource_utils import run_result_path, resource_path

SIMILARITY = 'cosine_similarity'

EMBEDDING_CALCULATOR = ClipEmbeddingCalculatorImpl()
IMAGE_CONVERTOR = ParametricToImageConvertor()


def _get_counterfactuals(generator: CounterfactualsGenerator) -> pd.DataFrame:
    try:
        return generator.sample_with_weights(150,
                                             1,
                                             1,
                                             1,
                                             1,
                                             bonus_objectives_weights=np.array([10]).reshape((1, 1)),
                                             include_dataset=False)
    except ValueError as e:
        print(f"MCD failed to sample. Returning empty dataframe...")
        print_exception(e)
        return pd.DataFrame()


def _attempt_sample_and_render(generator: CounterfactualsGenerator, result_dir: str, batch_number: int,
                               target_embedding: np.ndarray):
    counterfactuals = to_full_dataframe(_get_counterfactuals(generator))
    batch_result_dir = _make_batch_dir(batch_number, result_dir)
    counterfactuals.to_csv(path_or_buf=os.path.join(batch_result_dir, "counterfactuals.csv"))
    closest_counterfactuals = _get_closest(counterfactuals, target_embedding)
    for cf_index in closest_counterfactuals.index:
        rendering_result = IMAGE_CONVERTOR.to_image(closest_counterfactuals.loc[cf_index])
        _save_rendering_result(cf_index, rendering_result, batch_result_dir)


def _get_closest(counterfactuals, target_embedding: np.ndarray):
    sample_size = min(len(counterfactuals), 5)
    counterfactuals[SIMILARITY] = get_cosine_similarity(PREDICTOR.predict(counterfactuals), target_embedding)
    closest_cfs = counterfactuals.sort_values(by=SIMILARITY, ascending=False)[:sample_size]
    print(f"Closest counterfactuals found: {closest_cfs[SIMILARITY]}")
    return closest_cfs


def _make_batch_dir(batch_number: int, result_dir: str):
    batch_result_dir = os.path.join(result_dir, f"_batch_{batch_number}")
    os.makedirs(batch_result_dir, exist_ok=True)
    return batch_result_dir


def _save_rendering_result(cf_index, rendering_result: RenderingResult, batch_result_dir):
    with open(os.path.join(batch_result_dir, f"bike_{cf_index}.txt"), "w") as text_file:
        text_file.write(rendering_result.bike_xml)
    with open(os.path.join(batch_result_dir, f"bike_{cf_index}.svg"), "wb") as image_file:
        image_file.write(rendering_result.image)


def _build_run_id(target_text: str):
    return target_text.lower().replace(' ', "-") + "-" + (str(uuid.uuid4().fields[-1])[:5])


def run_counterfactual_generation_task(run_description,
                                       total_generations,
                                       number_of_batches,
                                       target_embedding):
    assert total_generations > number_of_batches
    assert total_generations % number_of_batches == 0

    run_id = _build_run_id(run_description)
    results_dir = run_result_path(run_id)
    os.makedirs(results_dir, exist_ok=False)

    batch_size = total_generations // number_of_batches

    generator = build_generator(pop_size=100,
                                initialize_from_dataset=True,
                                target_embedding=target_embedding,
                                maximum_cosine_distance=0.7)

    for i in range(1, number_of_batches + 1):
        cumulative_gens = batch_size * i
        generator.generate(n_generations=cumulative_gens,
                           seed=42)
        # generator.save(os.path.join(results_dir, f"generator_{cumulative_gens}"))
        _attempt_sample_and_render(generator, results_dir, i, target_embedding)


if __name__ == "__main__":
    n_generations = 400
    run_counterfactual_generation_task(
        run_description=f"blue mountain bike",
        total_generations=n_generations,
        number_of_batches=8,
        target_embedding=EMBEDDING_CALCULATOR.from_text('a blue mountain bike'))
