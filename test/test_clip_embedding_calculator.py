import os.path
import unittest

import numpy as np

from embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl

BIKE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "resources", "black_bicycle.jpeg")


class ClipEmbeddingCalculatorTest(unittest.TestCase):
    def setUp(self):
        self.embedding_calculator = ClipEmbeddingCalculatorImpl()

    def test_embedding_from_text(self):
        self.assertValidEmbedding(self.embedding_calculator.from_text("Black bike"))

    def test_embedding_from_image_path(self):
        self.assertValidEmbedding(self.embedding_calculator.from_image_path(BIKE_IMAGE_PATH))

    def assertValidEmbedding(self, embedding):
        self.assertTrue(type(embedding) is np.ndarray)
        self.assertEqual((1, 512), embedding.shape)
