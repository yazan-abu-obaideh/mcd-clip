import os.path
import unittest

import pandas as pd
import numpy.testing as np_test
from sklearn.metrics import r2_score

from mcd_clip.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from mcd_clip.bike_embedding.embedding_comparator import get_cosine_similarity
from mcd_clip.bike_embedding.embedding_predictor import EmbeddingPredictor


def get_test_resource_path(resource_name) -> str:
    return str(os.path.join(os.path.dirname(__file__), "resources", resource_name))


class EmbeddingPredictorTest(unittest.TestCase):
    def setUp(self):
        self.embedding_predictor = EmbeddingPredictor()
        self.embeddings = pd.read_csv(get_test_resource_path("subset_embeddings.csv"), index_col=0)
        self.parameters = pd.read_csv(get_test_resource_path("subset_parametric.csv"), index_col=0)

    def test_r2(self):
        calc = ClipEmbeddingCalculatorImpl()
        target = calc.from_text("A green bicycle with thick wheels")
        print(r2_score(
            get_cosine_similarity(self.embeddings.values, target),
            get_cosine_similarity(self.embedding_predictor.predict(self.parameters), target)
        ))

    def test_predictor(self):
        predictions = self.embedding_predictor.predict(self.parameters)
        self.assertEqual(predictions.shape, (len(self.parameters), 512))

    def test_order_does_not_matter(self):
        data = self.parameters.iloc[0:1]
        predictions_og = self.embedding_predictor.predict(data)
        first_column = list(data.columns)[0]
        modified_data = data.drop(labels=[first_column], axis=1)
        modified_data[first_column] = data[first_column]
        reordered_predictions = self.embedding_predictor.predict(modified_data)
        np_test.assert_array_equal(
            predictions_og,
            reordered_predictions
        )

    @unittest.skip
    def test_handles_nan(self):
        pass
