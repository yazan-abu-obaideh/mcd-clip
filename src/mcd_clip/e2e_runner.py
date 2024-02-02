import os
import uuid

import pandas as pd
from decode_mcd import CounterfactualsGenerator

from mcd_clip.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl

from mcd_clip.bike_embedding.embedding_similarity_optimizer import build_generator, to_full_dataframe
from mcd_clip.bike_rendering.parametric_to_image_convertor import ParametricToImageConvertor, RenderingResult

from mcd_clip.resource_utils import run_result_path

EMBEDDING_CALCULATOR = ClipEmbeddingCalculatorImpl()
IMAGE_CONVERTOR = ParametricToImageConvertor()


def _get_counterfactuals(generator: CounterfactualsGenerator) -> pd.DataFrame:
    try:
        return generator.sample_with_weights(1000, 1, 1, 1, 1)
    except ValueError:
        print("MCD failed to sample. Returning empty dataframe...")
        return pd.DataFrame()


def _attempt_sample_and_render(generator: CounterfactualsGenerator, result_dir: str, batch_number: int):
    counterfactuals = to_full_dataframe(_get_counterfactuals(generator))
    for cf_index in counterfactuals.index:
        rendering_result = IMAGE_CONVERTOR.to_image(counterfactuals.loc[cf_index])
        _save_rendering_result(cf_index, rendering_result, result_dir, batch_number)


def _save_rendering_result(cf_index, rendering_result: RenderingResult, result_dir, batch_number: int):
    batch_result_dir = os.path.join(result_dir, f"_batch_{batch_number}")
    os.makedirs(batch_result_dir, exist_ok=True)
    with open(os.path.join(batch_result_dir, f"bike_{cf_index}.txt"), "w") as text_file:
        text_file.write(rendering_result.bike_xml)
    with open(os.path.join(batch_result_dir, f"bike_{cf_index}.png"), "wb") as image_file:
        image_file.write(rendering_result.image)


def run_counterfactual_generation_task():
    target_text = "A cool green bike"
    total_generations = 3000
    number_of_batches = 15
    run_id = str(uuid.uuid4().fields[-1])[:5]

    assert total_generations > number_of_batches
    assert total_generations % number_of_batches == 0

    batch_size = total_generations // number_of_batches

    generator = build_generator(pop_size=100,
                                initialize_from_dataset=True,
                                target_embedding=EMBEDDING_CALCULATOR.from_text(target_text))

    results_dir = run_result_path(run_id)
    os.makedirs(results_dir, exist_ok=False)

    for i in range(1, number_of_batches + 1):
        cumulative_gens = batch_size * i
        generator.generate(n_generations=cumulative_gens,
                           seed=42)
        generator.save(os.path.join(results_dir, f"generator_{cumulative_gens}"))
        _attempt_sample_and_render(generator, results_dir, i)


if __name__ == "__main__":
    run_counterfactual_generation_task()
