import unittest

import numpy as np

from clip_embedding_calculator import ClipEmbeddingCalculatorImpl


class ClipEmbeddingCalculatorTest(unittest.TestCase):
    def setUp(self):
        self.embedding_calculator = ClipEmbeddingCalculatorImpl()

    def test_embedding_from_text(self):
        embedding = self.embedding_calculator.from_text("Black bike")
        self.assertTrue(type(embedding) is np.ndarray)
        self.assertEqual((1, 512), embedding.shape)
