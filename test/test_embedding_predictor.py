import os.path
import unittest

import pandas as pd

from embedding_predictor import EmbeddingPredictor


def get_test_resource_path(resource_name) -> str:
    return str(os.path.join(os.path.dirname(__file__), "resources", resource_name))


class EmbeddingPredictorTest(unittest.TestCase):
    def setUp(self):
        self.embedding_predictor = EmbeddingPredictor()
        self.embeddings = pd.read_csv(get_test_resource_path("subset_embeddings.csv"), index_col="Unnamed: 0")
        self.parameters = pd.read_csv(get_test_resource_path("subset_parametric.csv"), index_col="Unnamed: 0")

    def test_predictor(self):
        predictions = self.embedding_predictor.predict(self.parameters)
        print(predictions)
