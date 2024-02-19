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
from mcd_clip.datasets.combined_datasets import CombinedDataset, map_combined_datatypes, \
    OriginalCombinedDataset
from mcd_clip.optimization.embedding_similarity_optimizer import predict_from_partial_dataframe, CONSTANT_COLUMNS
from mcd_clip.resource_utils import run_result_path, resource_path
from mcd_clip.structural.load_data import load_augmented_framed_dataset
from mcd_clip.structural.structural_predictor import StructuralPredictor
from test_fit_analysis import SAMPLE_RIDER

EMBEDDING_CALCULATOR = ClipEmbeddingCalculatorImpl()
ORIGINAL_COMBINED = OriginalCombinedDataset()

_AVG_GOWER_INDEX = -1

_CHANGED_FEATURE_INDEX = -2

_GOWER_INDEX = -3


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
        return f"TextEmbeddingTarget: [{str(self._text_target)}]"

    def get_embedding(self) -> np.ndarray:
        return self._embedding


class ImageEmbeddingTarget(EmbeddingTarget):
    def __init__(self, image_path: str):
        self._image_path = image_path
        self._embedding = EMBEDDING_CALCULATOR.from_image_path(image_path)

    def __str__(self):
        return f"ImageEmbeddingTarget: [{str(os.path.split(self._image_path)[-1])}]"

    def get_embedding(self) -> np.ndarray:
        return self._embedding


class CombinedOptimizer:
    def __init__(self,
                 design_targets: DesignTargets,
                 target_embeddings: List[EmbeddingTarget],
                 extra_bonus_objectives: List[str]):
        self._target_embeddings = target_embeddings
        _, y, self._x_scaler, self._y_scaler = load_augmented_framed_dataset()
        self._design_targets = design_targets
        self._structural_predictor = StructuralPredictor()
        self._extra_bonus_objectives = extra_bonus_objectives or []
        self._starting_dataset = self._build_starting_dataset()
        self.structural_enabled = ('Model Mass' in design_targets.get_all_constrained_labels() or
                                   "Sim 1 Safety Factor (Inverted)" in design_targets.get_all_constrained_labels())
        self.starting_design = None

    def set_starting_design_by_index(self, design_index: str):
        self.starting_design = self._get_starting_design(design_index)

    def set_starting_design(self, design: pd.DataFrame):
        self.starting_design = design

    def _log_starting_performance(self):
        predictions = self.predict(self.starting_design)
        for column in predictions.columns:
            print(f"Starting design has in column {column} value {predictions[column].values[0]}")

    def build_generator(self,
                        gower_on: bool = True,
                        average_gower_on: bool = True,
                        changed_feature_on: bool = True,
                        use_empty_repair: bool = False,
                        validation_functions: List[Callable] = None) -> CounterfactualsGenerator:

        if self.starting_design is None:
            raise Exception("Must set starting design before building the generator")

        self._log_starting_performance()
        validation_functions = validation_functions or []
        print(f"Number of validation functions is {len(validation_functions)}")
        data_package = DataPackage(
            features_dataset=self._starting_dataset.get_combined(),
            predictions_dataset=self.predict(self._starting_dataset),
            query_x=self.starting_design.get_combined(),
            design_targets=self._design_targets,
            datatypes=map_combined_datatypes(self._starting_dataset.get_combined()),
            bonus_objectives=self.distance_columns() + self._extra_bonus_objectives
        )
        problem = MultiObjectiveProblem(
            data_package=data_package,
            prediction_function=lambda d: self.predict(CombinedDataset(
                pd.DataFrame(d, columns=self._starting_dataset.get_combined().columns))),
            constraint_functions=validation_functions
        )
        problem.set_desired_scores(
            gower=gower_on,
            average_gower=average_gower_on,
            change_feature_ratio=changed_feature_on
        )
        generator = CounterfactualsGenerator(
            problem=problem,
            pop_size=100,
            initialize_from_dataset=True,
        )
        generator.use_empty_repair(use_empty_repair)
        return generator

    def _build_starting_dataset(self) -> CombinedDataset:
        original_dataset = ORIGINAL_COMBINED.get_combined_dataset()
        return CombinedDataset.build_from_three(framed_style=original_dataset.get_as_framed(),
                                                clips_style=original_dataset.get_as_clips().drop(
                                                    columns=CONSTANT_COLUMNS),
                                                bike_fit_style=original_dataset.get_as_bike_fit())

    def predict(self, designs: CombinedDataset) -> pd.DataFrame:
        result = pd.DataFrame(index=designs.get_combined().index)
        if self.structural_enabled:
            self._predict_structural(result, designs)
        self._predict_embedding_distances(result, designs)
        result = self._add_fit_measure(designs, result, calculate_drag)
        result = self._add_fit_measure(designs, result, calculate_angles)
        result = self._drop_irrelevant_columns(result)
        assert len(result) == len(designs.get_combined()), "Concat failed!"
        self._log_progress(result)
        return result

    def _get_relevant_columns(self) -> List[str]:
        return list(
            self._design_targets.get_all_constrained_labels()) + self._extra_bonus_objectives + self.distance_columns()

    def _predict_structural(self, result: pd.DataFrame,
                            designs: CombinedDataset) -> None:
        structural_predictions = self._structural_predictor.predict_unscaled(
            designs.get_as_framed(), self._x_scaler, self._y_scaler)
        for c in structural_predictions.columns:
            result[c] = structural_predictions[c]

    def _drop_irrelevant_columns(self, result: pd.DataFrame) -> pd.DataFrame:
        return result.drop(columns=[c for c in result.columns if c
                                    not in self._get_relevant_columns()])

    def _predict_embedding_distances(self, result: pd.DataFrame, designs: CombinedDataset) -> None:
        designs_clips = designs.get_as_clips()
        for idx in range(len(self._target_embeddings)):
            target = self._target_embeddings[idx].get_embedding()
            result[distance_column_name(idx)] = predict_from_partial_dataframe(designs_clips, target)

    def _add_fit_measure(self,
                         designs: CombinedDataset,
                         older_results: pd.DataFrame,
                         evaluation_function: Callable[[np.ndarray, np.ndarray], pd.DataFrame]):
        evaluation_result = evaluation_function(designs.get_as_bike_fit().values, SAMPLE_RIDER)
        evaluation_result.index = older_results.index
        return pd.concat([older_results, evaluation_result], axis=1)

    def distance_columns(self) -> List[str]:
        return [distance_column_name(i) for i in range(len(self._target_embeddings))]

    def _log_progress(self, result: pd.DataFrame):
        for column in result.columns:
            result_column_ = result[column]
            print(f"{column} | min {round(np.min(result_column_), 2)}, max [{round(np.max(result_column_), 2)}],"
                  f"  average [{round(np.average(result_column_), 2)}]")

    def _get_starting_design(self, design_index: str) -> CombinedDataset:
        return CombinedDataset(
            pd.DataFrame.from_records(
                [self._starting_dataset.get_combined().loc[design_index].to_dict()]
            )
        )


def _to_scores_dataframe(scores: np.ndarray):
    data_frame = pd.DataFrame()
    data_frame['gower_distance'] = scores[:, _GOWER_INDEX]
    data_frame['avg_gower_distance'] = scores[:, _AVG_GOWER_INDEX]
    data_frame['changed_feature_ratio'] = scores[:, _CHANGED_FEATURE_INDEX]
    return data_frame


def run_generation_task() -> CounterfactualsGenerator:
    target_embeddings = [
        TextEmbeddingTarget(text_target='A futuristic black cyberpunk-style road racing bicycle'),
        ImageEmbeddingTarget(image_path=resource_path('mtb.png'))
    ]
    design_targets = DesignTargets(
        continuous_targets=[
            ContinuousTarget('Sim 1 Safety Factor (Inverted)', lower_bound=0, upper_bound=1),
            ContinuousTarget('Model Mass', lower_bound=0, upper_bound=5.5),
            BACK_TARGET,
            ARMPIT_WRIST_TARGET,
            KNEE_TARGET,
            # AERODYNAMIC_DRAG_TARGET
        ])

    bonus_objectives = ["Model Mass", AERODYNAMIC_DRAG_TARGET.label]

    optimizer = CombinedOptimizer(
        design_targets=design_targets,
        target_embeddings=target_embeddings,
        extra_bonus_objectives=bonus_objectives
    )

    optimizer.set_starting_design_by_index('3728')

    generator = optimizer.build_generator()

    number_of_batches = 4
    batch_size = 75

    run_id = 'full-scores-fit-bikes-' + str(uuid.uuid4().fields[-1])[:5]
    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)
    _save_metadata(batch_size, number_of_batches, run_dir, target_embeddings)

    for i in range(1, number_of_batches + 1):
        cumulative_gens = batch_size * i
        generator.generate(n_generations=cumulative_gens, seed=23)
        sampled = generator.sample_with_weights(num_samples=1000, avg_gower_weight=1, gower_weight=1,
                                                cfc_weight=1, diversity_weight=0.1)
        result = pd.concat(
            [sampled,
             get_scores_dataframe(generator, sampled),
             optimizer.predict(CombinedDataset(sampled))
             ],
            axis=1
        )
        assert len(sampled) == len(result), "concat failed"
        result.to_csv(os.path.join(run_dir, f'batch_{i}_cfs.csv'))
    return generator


def get_scores_dataframe(generator: CounterfactualsGenerator, sampled: pd.DataFrame):
    return _to_scores_dataframe(_get_scores_array(generator, sampled))


def _save_metadata(batch_size, number_of_batches, run_dir, target_embeddings):
    with open(os.path.join(run_dir, 'metadata.txt'), 'w') as file:
        file.write(
            f"Target embeddings: {target_embeddings}\n"
            + f"batches: {number_of_batches}. Batch size: {batch_size}"
        )


def _get_scores_array(generator: CounterfactualsGenerator, sampled: pd.DataFrame):
    result_dict = {}
    generator._problem._evaluate(sampled.values, result_dict)
    return result_dict['F']


if __name__ == '__main__':
    run_generation_task()
