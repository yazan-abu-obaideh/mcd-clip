import unittest

from bike_embedding import clip_embedding_calculator
from bike_embedding.embedding_similarity_optimizer import get_features, map_datatypes, do_problem


class EmbeddingSimilarityOptimizerTest(unittest.TestCase):
    def test_get_features(self):
        print(get_features())

    def test_map_datatypes(self):
        print(map_datatypes())

    def test_no_errors_thrown_when_run(self):
        embedding_calculator = clip_embedding_calculator.ClipEmbeddingCalculatorImpl()
        do_problem(embedding_calculator.from_text("Black bicycle").reshape((512,)),
                   pop_size=5, n_generations=3
                   )
