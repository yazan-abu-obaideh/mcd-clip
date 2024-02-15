import unittest

from pymoo.core.variable import Real, Variable, Choice, Integer

from mcd_clip.bike_embedding import clip_embedding_calculator
from mcd_clip.optimization.embedding_similarity_optimizer import get_features, map_datatypes, optimize_similarity


class EmbeddingSimilarityOptimizerTest(unittest.TestCase):
    def test_get_features(self):
        print(get_features())

    def test_map_datatypes(self):
        types_string = ""
        mapped_datatypes = map_datatypes()
        for t in mapped_datatypes:
            types_string += self._t_to_string(t)
        self.assertEqual(len(types_string), 89)
        self.assertEqual(
            "RRRRRRRRRRRRRRRRRRRRRRRCCRRRRRRRCRRRRCCIICIIIIRRRRRICCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            types_string
        )

    def _t_to_string(self, t: Variable):
        if type(t) is Real:
            return 'R'
        if type(t) is Integer:
            return 'I'
        if type(t) is Choice:
            return 'C'

    def test_no_errors_thrown_when_run(self):
        embedding_calculator = clip_embedding_calculator.ClipEmbeddingCalculatorImpl()
        optimize_similarity(embedding_calculator.from_text("Black bicycle"),
                            pop_size=5, n_generations=3
                            )
