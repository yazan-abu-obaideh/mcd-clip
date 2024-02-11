import os
import uuid
from typing import List

import numpy as np
import pandas as pd
from decode_mcd import DataPackage, DesignTargets, CounterfactualsGenerator, MultiObjectiveProblem, ContinuousTarget

from mcd_clip.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from mcd_clip.bike_embedding.embedding_similarity_optimizer import CONSTANT_COLUMNS, predict_cosine_distance
from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.biked.structural_predictor import StructuralPredictor
from mcd_clip.combined_optimization.combined_datasets import CombinedDataset, map_combined_datatypes
from mcd_clip.resource_utils import resource_path, run_result_path

STRUCTURAL_PREDICTOR = StructuralPredictor()
EMBEDDING_CALCULATOR = ClipEmbeddingCalculatorImpl()

framed_x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
framed_x = pd.DataFrame(x_scaler.inverse_transform(framed_x),
                        columns=framed_x.columns,
                        index=framed_x.index)
clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
clips.index = [str(idx) for idx in clips.index]

intersection = set(framed_x.index).intersection(set(clips.index))

framed_x = framed_x.loc[intersection]
clips = clips.loc[intersection]

print(len(framed_x))
print(len(clips))


class CombinedOptimizer:
    def __init__(self, target_embeddings: List[np.ndarray]):
        self._target_embeddings = target_embeddings

    def predict(self, designs: CombinedDataset) -> pd.DataFrame:
        structural_predictions = STRUCTURAL_PREDICTOR.predict(designs.get_as_framed())
        designs_clips = designs.get_as_clips()
        embedding_predictions = pd.DataFrame(
            columns=[f'embedding_distance_{i}' for i in range(1, len(self._target_embeddings) + 1)],
            index=designs_clips.index)
        for idx in range(len(self._target_embeddings)):
            embedding_predictions[f'embedding_distance_{idx + 1}'] = predict_cosine_distance(designs_clips,
                                                                                             self._target_embeddings[
                                                                                                 idx])

        concat = pd.concat([structural_predictions, embedding_predictions], axis=1)
        return pd.DataFrame(concat, columns=['Model Mass', 'Sim 1 Safety Factor (Inverted)'] + list(
            embedding_predictions.columns))


if __name__ == '__main__':
    optimizer = CombinedOptimizer(
        target_embeddings=[
            EMBEDDING_CALCULATOR.from_text('A futuristic black cyberpunk-style road racing bicycle'),
            EMBEDDING_CALCULATOR.from_image_path(resource_path('mtb.png')),
        ]
    )
    clips.drop(columns=CONSTANT_COLUMNS, inplace=True)
    original_combined = CombinedDataset.build_from_both(framed_style=framed_x, clips_style=clips)
    data_package = DataPackage(
        features_dataset=original_combined.get_combined(),
        predictions_dataset=optimizer.predict(original_combined),
        query_x=original_combined.get_combined().iloc[0:1],
        design_targets=DesignTargets(
            continuous_targets=[
                ContinuousTarget('Model Mass', lower_bound=0, upper_bound=2),
                ContinuousTarget('Sim 1 Safety Factor (Inverted)', lower_bound=0, upper_bound=1),
            ]
        ),
        datatypes=map_combined_datatypes(original_combined.get_combined()),
        bonus_objectives=[
            'embedding_distance_1',
            'embedding_distance_2',
        ]
    )

    problem = MultiObjectiveProblem(
        data_package=data_package,
        prediction_function=lambda d: optimizer.predict(CombinedDataset(
            pd.DataFrame(d, columns=original_combined.get_combined().columns))),
        constraint_functions=[]
    )

    generator = CounterfactualsGenerator(
        problem=problem,
        pop_size=100,
        initialize_from_dataset=True,
    )

    number_of_batches = 10
    batch_size = 100

    run_dir = run_result_path(str(uuid.uuid4()))
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, 'metadata.txt'), 'w') as file:
        file.write(
            f"Target embeddings: A futuristic black cyberpunk-style road racing bicycle | mtb.png\n"
            + f"batches: {number_of_batches}. Batch size: {batch_size}"
        )

    for i in range(1, number_of_batches + 1):
        cumulative_gens = batch_size * i
        generator.generate(n_generations=cumulative_gens,
                           seed=42)
        sampled = generator.sample_with_weights(num_samples=100, avg_gower_weight=1, gower_weight=1, cfc_weight=1,
                                                diversity_weight=1)
        sampled: pd.DataFrame
        sampled.to_csv(os.path.join(run_dir, f'batch_{i}_cfs.csv'))
