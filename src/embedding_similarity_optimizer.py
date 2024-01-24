import os.path

import numpy as np
import pandas as pd
from decode_mcd import DesignTargets, DataPackage, MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget

import clip_embedding_calculator
from datatypes_mapper import map_column
from embedding_comparator import get_cosine_similarity
from embedding_predictor import EmbeddingPredictor

PREDICTOR = EmbeddingPredictor()

CONSTANT_COLUMNS = ['Wall thickness Bottom Bracket', 'Wall thickness Top tube',
                    'Wall thickness Head tube', 'Wall thickness Down tube']


def get_features():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "resources", "clip_sBIKED_processed.csv"),
                       index_col=0)


def get_labels(target_embedding: np.ndarray):
    predictions = PREDICTOR.predict(get_features())
    return 1 - get_cosine_similarity(predictions, target_embedding)


def predict(designs, target_embedding):
    designs_copy = designs.copy().astype("float64")
    for column in CONSTANT_COLUMNS:
        designs_copy[column] = get_features()[column].mean()
    return 1 - get_cosine_similarity(PREDICTOR.predict(designs_copy), target_embedding)


def map_datatypes():
    datatypes = []
    features = get_features()
    for column_name in list(features.columns):
        if column_name not in CONSTANT_COLUMNS:
            datatypes.append(map_column(features[column_name]))
    return datatypes


def do_problem(target_embedding: np.ndarray, pop_size=1000, n_generations=30, initialize_from_dataset=False):
    features_dataset = get_features().drop(columns=CONSTANT_COLUMNS)
    data_package = DataPackage(features_dataset=features_dataset,
                               predictions_dataset=pd.DataFrame(get_labels(target_embedding),
                                                                columns=["cosine_distance"],
                                                                index=features_dataset.index),
                               query_x=get_features().iloc[0:1].drop(columns=CONSTANT_COLUMNS),
                               design_targets=DesignTargets([ContinuousTarget(label="cosine_distance",
                                                                              lower_bound=0,
                                                                              upper_bound=0.2)]),
                               datatypes=map_datatypes(),
                               bonus_objectives=["cosine_distance"])

    problem = MultiObjectiveProblem(data_package=data_package,
                                    prediction_function=lambda design: predict(design, target_embedding),
                                    constraint_functions=[])

    generator = CounterfactualsGenerator(problem=problem,
                                         pop_size=pop_size,
                                         initialize_from_dataset=initialize_from_dataset)

    generator.generate(n_generations=n_generations)
    return generator.sample_with_dtai(num_samples=10, gower_weight=1,
                                      avg_gower_weight=1, cfc_weight=1,
                                      diversity_weight=50)


if __name__ == "__main__":
    embedding_calculator = clip_embedding_calculator.ClipEmbeddingCalculatorImpl()
    cfs = do_problem(embedding_calculator.from_text("Black bicycle").reshape((512,)))
    with open("result.csv", "w") as file:
        cfs.to_csv(file)