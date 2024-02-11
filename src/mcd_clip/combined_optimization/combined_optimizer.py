import os
import uuid
from typing import List

import numpy as np
import pandas as pd
from decode_mcd import DataPackage, DesignTargets, CounterfactualsGenerator, MultiObjectiveProblem, ContinuousTarget
from decode_mcd.design_targets import McdTarget

from mcd_clip.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from mcd_clip.bike_embedding.embedding_similarity_optimizer import CONSTANT_COLUMNS, predict_cosine_distance
from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.biked.structural_predictor import StructuralPredictor
from mcd_clip.combined_optimization.combined_datasets import CombinedDataset, map_combined_datatypes
from mcd_clip.resource_utils import resource_path, run_result_path

STRUCTURAL_PREDICTOR = StructuralPredictor()
EMBEDDING_CALCULATOR = ClipEmbeddingCalculatorImpl()


# framed_x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
# framed_x = pd.DataFrame(x_scaler.inverse_transform(framed_x),
#                         columns=framed_x.columns,
#                         index=framed_x.index)
# clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
# clips.index = [str(idx) for idx in clips.index]
#
# intersection = set(framed_x.index).intersection(set(clips.index))
#
# framed_x = framed_x.loc[intersection]
# clips = clips.loc[intersection]


class EmbeddingTarget:
    def get_embedding(self):
        pass


class TextEmbeddingTarget(EmbeddingTarget):
    def __init__(self, text_target: str):
        self._text_target = text_target
        self._embedding = EMBEDDING_CALCULATOR.from_text(self._text_target)

    def __str__(self):
        return f"TextEmbeddingTarget: [{self._text_target}]"

    def get_embedding(self):
        return self._embedding


class ImageEmbeddingTarget(EmbeddingTarget):
    def __init__(self, image_path: str):
        self._image_path = image_path
        self._embedding = EMBEDDING_CALCULATOR.from_image_path(image_path)

    def __str__(self):
        return f"ImageEmbeddingTarget: [{os.path.split(self._image_path)[-1]}]"

    def get_embedding(self):
        super().get_embedding()


class CombinedOptimizer:
    def __init__(self,
                 structural_targets: List[McdTarget],
                 target_embeddings: List[EmbeddingTarget]):
        self._target_embeddings = target_embeddings
        _, y, self._x_scaler, self._y_scaler = load_augmented_framed_dataset()
        self.structural_targets = structural_targets
        as_text = [t.label for t in self.structural_targets]
        self._structural_predictor = StructuralPredictor(exclude_predictors=[c for c in y.columns if c not in as_text])

    def predict(self, designs: CombinedDataset) -> pd.DataFrame:
        structural_predictions = STRUCTURAL_PREDICTOR.predict_unscaled(designs.get_as_framed(),
                                                                       self._x_scaler,
                                                                       self._y_scaler)
        designs_clips = designs.get_as_clips()
        embedding_predictions = pd.DataFrame(
            columns=[f'embedding_distance_{i}' for i in range(1, len(self._target_embeddings) + 1)],
            index=designs_clips.index)
        for idx in range(len(self._target_embeddings)):
            embedding_predictions[f'embedding_distance_{idx + 1}'] = predict_cosine_distance(designs_clips,
                                                                                             self._target_embeddings[
                                                                                                 idx])

        concat = pd.concat([structural_predictions, embedding_predictions], axis=1)

        result = pd.DataFrame(concat, columns=['Model Mass', 'Sim 1 Safety Factor (Inverted)'] + list(
            embedding_predictions.columns))
        self._log_nans(result.astype('float64'))
        return result

    def _get_x_scaler(self):
        return load_augmented_framed_dataset()[2]

    def _log_nans(self, result):
        nan_columns = [c for c in result.columns if result[c].isna().any()]
        if nan_columns:
            print(f"WARNING: found nan columns {nan_columns}")


if __name__ == '__main__':
    optimizer = CombinedOptimizer(
        target_embeddings=[
            TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
            ImageEmbeddingTarget(image_path=resource_path('mtb.png')),
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
        generator.generate(n_generations=cumulative_gens)
        sampled = generator.sample_with_weights(num_samples=100, avg_gower_weight=1, gower_weight=1, cfc_weight=1,
                                                diversity_weight=1)
        sampled: pd.DataFrame
        sampled.to_csv(os.path.join(run_dir, f'batch_{i}_cfs.csv'))
