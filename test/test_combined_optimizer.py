import unittest

from decode_mcd import DesignTargets, ContinuousTarget

from mcd_clip.optimization.combined_optimizer import CombinedOptimizer, TextEmbeddingTarget


class CombinedOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = CombinedOptimizer(
            design_targets=DesignTargets(
                continuous_targets=[ContinuousTarget(label='Model Mass', lower_bound=0, upper_bound=2)]
            ),
            target_embeddings=[TextEmbeddingTarget('A yellow mountain bike')]
        )

    def test_generator_runs(self):
        generator = self.optimizer.build_generator()
        generator.generate(
            n_generations=3,
        )
        generator.sample_with_weights(
            10, 1,
            1, 1, 1,
        )
