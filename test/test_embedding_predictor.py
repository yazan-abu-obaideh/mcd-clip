import os.path
import unittest

import pandas as pd
import torch

from embedding_predictor import EmbeddingPredictor


def get_test_resource_path(resource_name) -> str:
    return str(os.path.join(os.path.dirname(__file__), "resources", resource_name))


class EmbeddingPredictorTest(unittest.TestCase):
    def setUp(self):
        self.embedding_predictor = EmbeddingPredictor()
        self.embeddings = pd.read_csv(get_test_resource_path("subset_embeddings.csv"), index_col="Unnamed: 0")
        self.parameters = pd.read_csv(get_test_resource_path("subset_parametric.csv"), index_col="Unnamed: 0")

    def test_predictor(self):
        predictor_input = torch.tensor(self.parameters.values, dtype=torch.float32)
        predictions = self.embedding_predictor.predict(predictor_input)
        print(predictions.shape)
        print(predictions.detach().numpy())
