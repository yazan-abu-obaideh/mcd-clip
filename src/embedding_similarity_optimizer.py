import os.path

import numpy as np
import pandas as pd
from decode_mcd import DesignTargets, DataPackage, MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget

from datatypes_mapper import map_column
from embedding_comparator import get_cosine_similarity
from embedding_predictor import EmbeddingPredictor

PREDICTOR = EmbeddingPredictor()


def get_features():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "resources", "clip_sBIKED_processed.csv"),
                       index_col=0)


def get_labels(target_embedding: np.ndarray):
    predictions = PREDICTOR.predict(get_features())
    return 1 - get_cosine_similarity(predictions, target_embedding)


def predict(designs, target_embedding):
    return 1 - get_cosine_similarity(PREDICTOR.predict(designs), target_embedding)


def map_datatypes():
    datatypes = []
    features = get_features()
    for column_name in list(features.columns):
        datatypes.append(map_column(features[column_name]))
    return datatypes


def do_problem(target_embedding: np.ndarray):
    data_package = DataPackage(features_dataset=get_features(),
                               predictions_dataset=pd.DataFrame(get_labels(target_embedding),
                                                                columns=["cosine_distance"]),
                               query_x=get_features().get(1),
                               design_targets=DesignTargets([ContinuousTarget(label="cosine_distance",
                                                                              lower_bound=0,
                                                                              upper_bound=1)]),
                               datatypes=map_datatypes(),
                               bonus_objectives=["cosine_distance"])

    problem = MultiObjectiveProblem(data_package=data_package,
                                    prediction_function=lambda design: predict(design, target_embedding),
                                    constraint_functions=[])

    generator = CounterfactualsGenerator(problem=problem,
                                         pop_size=10,
                                         initialize_from_dataset=False)

    generator.generate(n_generations=10)
    counterfactuals = generator.sample_with_dtai(num_samples=10, gower_weight=1,
                                                 avg_gower_weight=1, cfc_weight=1,
                                                 diversity_weight=50)
    print(counterfactuals)


if __name__ == "__main__":
    pass
