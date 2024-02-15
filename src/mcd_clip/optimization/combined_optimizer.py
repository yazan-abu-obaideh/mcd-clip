import os
import uuid
from abc import abstractmethod, ABCMeta
from typing import List, Callable

import numpy as np
import pandas as pd
from decode_mcd import DataPackage, DesignTargets, CounterfactualsGenerator, MultiObjectiveProblem, ContinuousTarget

from mcd_clip.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from mcd_clip.bike_rider_fit.fit_analysis.demoanalysis_wrapped import calculate_drag, calculate_angles
from mcd_clip.bike_rider_fit.fit_optimization import BACK_TARGET, ARMPIT_WRIST_TARGET, KNEE_TARGET, \
    AERODYNAMIC_DRAG_TARGET
from mcd_clip.optimization.embedding_similarity_optimizer import predict_from_partial_dataframe, CONSTANT_COLUMNS
from mcd_clip.structural.load_data import load_augmented_framed_dataset
from mcd_clip.structural.structural_predictor import StructuralPredictor
from mcd_clip.datasets.validations_lists import COMBINED_VALIDATION_FUNCTIONS
from mcd_clip.datasets.combined_datasets import CombinedDataset, map_combined_datatypes, \
    OriginalCombinedDataset
from mcd_clip.resource_utils import resource_path, run_result_path
from test_fit_analysis import SAMPLE_RIDER

EMBEDDING_CALCULATOR = ClipEmbeddingCalculatorImpl()
ORIGINAL_COMBINED = OriginalCombinedDataset()


def distance_column_name(column_index: int):
    return f"embedding_distance_{column_index + 1}"


class EmbeddingTarget(metaclass=ABCMeta):
    @abstractmethod
    def get_embedding(self) -> np.ndarray:
        pass


class TextEmbeddingTarget(EmbeddingTarget):
    def __init__(self, text_target: str):
        self._text_target = text_target
        self._embedding = EMBEDDING_CALCULATOR.from_text(self._text_target)

    def __str__(self):
        return f"TextEmbeddingTarget: [{self._text_target}]"

    def get_embedding(self) -> np.ndarray:
        return self._embedding


class ImageEmbeddingTarget(EmbeddingTarget):
    def __init__(self, image_path: str):
        self._image_path = image_path
        self._embedding = EMBEDDING_CALCULATOR.from_image_path(image_path)

    def __str__(self):
        return f"ImageEmbeddingTarget: [{os.path.split(self._image_path)[-1]}]"

    def get_embedding(self) -> np.ndarray:
        return self._embedding


class CombinedOptimizer:
    def __init__(self,
                 design_targets: DesignTargets,
                 target_embeddings: List[EmbeddingTarget]):
        self._target_embeddings = target_embeddings
        _, y, self._x_scaler, self._y_scaler = load_augmented_framed_dataset()
        self._design_targets = design_targets
        self._structural_predictor = StructuralPredictor()

    def build_generator(self) -> CounterfactualsGenerator:
        original_dataset = ORIGINAL_COMBINED.get_combined_dataset()
        starting_dataset = CombinedDataset.build_from_both(framed_style=original_dataset.get_as_framed(),
                                                           clips_style=original_dataset.get_as_clips().drop(
                                                               columns=CONSTANT_COLUMNS))
        data_package = DataPackage(
            features_dataset=starting_dataset.get_combined(),
            predictions_dataset=self.predict(starting_dataset),
            query_x=starting_dataset.get_combined().iloc[0:1],
            design_targets=self._design_targets,
            datatypes=map_combined_datatypes(starting_dataset.get_combined()),
            bonus_objectives=list(self._design_targets.get_all_constrained_labels()) + self.distance_columns()
        )
        problem = MultiObjectiveProblem(
            data_package=data_package,
            prediction_function=lambda d: self.predict(CombinedDataset(
                pd.DataFrame(d, columns=starting_dataset.get_combined().columns))),
            constraint_functions=COMBINED_VALIDATION_FUNCTIONS
        )
        generator = CounterfactualsGenerator(
            problem=problem,
            pop_size=100,
            initialize_from_dataset=True,
        )
        return generator

    def predict(self, designs: CombinedDataset) -> pd.DataFrame:
        structural_predictions = self._predict_structural(designs)
        embedding_predictions = self._predict_embedding_distances(designs)
        result = pd.concat([structural_predictions, embedding_predictions], axis=1)
        result = self._add_fit_measure(designs, result, calculate_drag)
        result = self._add_fit_measure(designs, result, calculate_angles)
        result = self._drop_irrelevant_columns(result)
        self._log_progress(result)
        return result

    def _predict_structural(self, designs: CombinedDataset) -> pd.DataFrame:
        return self._structural_predictor.predict_unscaled(designs.get_as_framed(),
                                                           self._x_scaler,
                                                           self._y_scaler)

    def _drop_irrelevant_columns(self, result: pd.DataFrame) -> pd.DataFrame:
        return result.drop(columns=[c for c in result.columns if c
                                    not in list(self._design_targets.get_all_constrained_labels())
                                    + self.distance_columns()])

    def _predict_embedding_distances(self, designs: CombinedDataset) -> pd.DataFrame:
        designs_clips = designs.get_as_clips()
        embedding_predictions = pd.DataFrame(columns=self.distance_columns(), index=designs_clips.index)
        for idx in range(len(self._target_embeddings)):
            target = self._target_embeddings[idx].get_embedding()
            embedding_predictions[distance_column_name(idx)] = predict_from_partial_dataframe(designs_clips, target)
        return embedding_predictions

    def _add_fit_measure(self,
                         designs: CombinedDataset,
                         result: pd.DataFrame,
                         evaluation_function: Callable[[np.ndarray, np.ndarray], pd.DataFrame]):
        angles = evaluation_function(designs.get_for_ergonomics().values, SAMPLE_RIDER)
        result = pd.concat([result, pd.DataFrame(angles, index=result.index)], axis=1)
        return result

    def distance_columns(self) -> List[str]:
        return [distance_column_name(i) for i in range(len(self._target_embeddings))]

    def _log_progress(self, result: pd.DataFrame):
        for column in result.columns:
            print(f"{column} average is now [{np.average(result[column])}]")


def run_generation_task() -> CounterfactualsGenerator:
    design_targets = DesignTargets(
        continuous_targets=[ContinuousTarget('Model Mass', lower_bound=0, upper_bound=3),
                            ContinuousTarget('Sim 1 Safety Factor (Inverted)',
                                             lower_bound=0, upper_bound=1),
                            BACK_TARGET,
                            ARMPIT_WRIST_TARGET,
                            # KNEE_TARGET,
                            AERODYNAMIC_DRAG_TARGET
                            ])
    target_embeddings = [
        TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
        ImageEmbeddingTarget(image_path=resource_path('mtb.png')),
    ]

    optimizer = CombinedOptimizer(
        design_targets=design_targets,
        target_embeddings=target_embeddings
    )

    generator = optimizer.build_generator()

    number_of_batches = 12
    batch_size = 50

    run_id = 'bike-fit-combined-run-' + str(uuid.uuid4().fields[-1])[:5]
    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, 'metadata.txt'), 'w') as file:
        file.write(
            f"Target embeddings: {target_embeddings}\n"
            + f"batches: {number_of_batches}. Batch size: {batch_size}"
        )

    for i in range(1, number_of_batches + 1):
        cumulative_gens = batch_size * i
        generator.generate(n_generations=cumulative_gens)
        sampled = generator.sample_with_weights(num_samples=1000, avg_gower_weight=1, gower_weight=1,
                                                cfc_weight=1, diversity_weight=1)
        sampled.to_csv(os.path.join(run_dir, f'batch_{i}_cfs.csv'))
        generator.save(os.path.join(run_dir, f'generator_{cumulative_gens}'))
    return generator


if __name__ == '__main__':
    run_generation_task()
