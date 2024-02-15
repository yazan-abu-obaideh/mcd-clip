from typing import Union

import numpy as np
import pandas as pd
from decode_mcd import DesignTargets, DataPackage, MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget

from mcd_clip.bike_embedding.embedding_comparator import get_cosine_distance
from mcd_clip.bike_embedding.embedding_predictor import EmbeddingPredictor
from mcd_clip.datasets.validations_lists import CLIPS_VALIDATION_FUNCTIONS
from mcd_clip.datasets.clips.datatypes_mapper import map_column
from mcd_clip.resource_utils import resource_path

PREDICTOR = EmbeddingPredictor()

CONSTANT_COLUMNS = ['Wall thickness Bottom Bracket', 'Wall thickness Top tube',
                    'Wall thickness Head tube', 'Wall thickness Down tube',
                    'Wall thickness Chain stay',
                    'Wall thickness Seat stay',
                    'Wall thickness Seat tube']


def get_features():
    return pd.read_csv(resource_path("clip_sBIKED_processed.csv"), index_col=0)


FEATURES = get_features()
TRIMMED_FEATURES = FEATURES.drop(columns=CONSTANT_COLUMNS)


def predict_from_partial_dataframe(designs, target_embedding):
    full_designs_df = to_full_dataframe(designs)
    return get_cosine_distance(PREDICTOR.predict(full_designs_df), target_embedding)


def to_full_dataframe(designs: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    designs_copy = pd.DataFrame(designs, columns=TRIMMED_FEATURES.columns)
    designs_copy = designs_copy.fillna(TRIMMED_FEATURES.mean())
    for column in CONSTANT_COLUMNS:
        designs_copy[column] = FEATURES[column].mean()
    return designs_copy


def map_datatypes():
    datatypes = []
    for column_name in list(FEATURES.columns):
        if column_name not in CONSTANT_COLUMNS:
            datatypes.append(map_column(FEATURES[column_name]))
    return datatypes


def build_generator(target_embedding: np.ndarray,
                    maximum_cosine_distance,
                    pop_size=1000,
                    initialize_from_dataset=False,
                    ):
    data_package = DataPackage(features_dataset=TRIMMED_FEATURES,
                               predictions_dataset=pd.DataFrame(
                                   get_cosine_distance(PREDICTOR.predict(FEATURES), target_embedding),
                                   columns=["cosine_distance"],
                                   index=TRIMMED_FEATURES.index),
                               query_x=TRIMMED_FEATURES.iloc[0:1],
                               design_targets=DesignTargets([ContinuousTarget(label="cosine_distance",
                                                                              lower_bound=0,
                                                                              upper_bound=maximum_cosine_distance)]),
                               datatypes=map_datatypes(),
                               bonus_objectives=["cosine_distance"])

    problem = MultiObjectiveProblem(data_package=data_package,
                                    prediction_function=lambda design:
                                    predict_from_partial_dataframe(design, target_embedding),
                                    constraint_functions=CLIPS_VALIDATION_FUNCTIONS)

    return CounterfactualsGenerator(problem=problem,
                                    pop_size=pop_size,
                                    initialize_from_dataset=initialize_from_dataset)


def optimize_similarity(target_embedding: np.ndarray,
                        pop_size=1000,
                        n_generations=30,
                        initialize_from_dataset=False,
                        sample_from_dataset=False,
                        maximum_cosine_distance=0.8):
    generator = build_generator(target_embedding,
                                maximum_cosine_distance,
                                pop_size,
                                initialize_from_dataset,
                                )
    generator.generate(n_generations=n_generations)
    return generator.sample_with_dtai(num_samples=1000, gower_weight=1,
                                      avg_gower_weight=1, cfc_weight=1,
                                      diversity_weight=1,
                                      include_dataset=sample_from_dataset)
