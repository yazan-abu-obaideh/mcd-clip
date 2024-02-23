import os
from datetime import datetime

import numpy as np
import pandas as pd
from decode_mcd import DesignTargets, ContinuousTarget

from mcd_clip.all_runs.ablation_runs import get_validity, render_some
from mcd_clip.datasets.combined_datasets import CombinedDataset
from mcd_clip.optimization.combined_optimizer import CombinedOptimizer
from mcd_clip.resource_utils import run_result_path


def run_with_specific_features(
        gower_on: bool,
        average_gower_on: bool,
        changed_feature_ratio_on: bool,
        use_empty_repair: bool,
        run_id_suffix: str,
        validation_functions
):
    GENERATIONS = 50
    BATCH_SIZE = 50
    BATCHES = GENERATIONS // BATCH_SIZE

    run_id = str(datetime.now().strftime('%m-%d--%H.%M.%S')) + '-ablation-template' + run_id_suffix

    optimizer = CombinedOptimizer(
        design_targets=DesignTargets(
            continuous_targets=[
                ContinuousTarget("Model Mass", 0, 7),
                ContinuousTarget("Sim 1 Safety Factor (Inverted)", 0, 1),
            ]
        ),
        target_embeddings=[],
        extra_bonus_objectives=['Model Mass', 'Sim 1 Safety Factor (Inverted)'],
    )
    optimizer.set_starting_design_by_index('1')
    generator = optimizer.build_generator(validation_functions=validation_functions,
                                          gower_on=gower_on,
                                          average_gower_on=average_gower_on,
                                          changed_feature_on=changed_feature_ratio_on,
                                          initialize_from_dataset=False,
                                          use_empty_repair=use_empty_repair,
                                          )

    run_dir = run_result_path(run_id)
    os.makedirs(run_dir, exist_ok=False)

    for i in range(1, BATCHES + 1):
        cumulative = i * BATCH_SIZE
        generator.generate(cumulative, seed=23)

        sampled = generator.sample_with_weights(num_samples=10,
                                                cfc_weight=3,
                                                gower_weight=3,
                                                avg_gower_weight=3,
                                                bonus_objectives_weights=np.array([1, 1]).reshape((1, 2)),
                                                diversity_weight=0.05,
                                                include_dataset=False)
        validity = get_validity(sampled)
        full_df = pd.concat([sampled, optimizer.predict(CombinedDataset(sampled)), validity], axis=1)
        assert len(full_df) == len(sampled)
        full_df.to_csv(os.path.join(run_dir, f"cfs_{i}.csv"))
        print(f"{run_id=}")
        print(f"Average CV/design for run {run_id_suffix} batch {i}: {np.mean(np.sum(validity, axis=1))}")
        print(f"Fraction of valid designs: {len(validity[np.sum(validity, axis=1) == 0])/len(validity)}")


if __name__ == '__main__':
    run_with_specific_features(
        gower_on=True,
        changed_feature_ratio_on=True,
        average_gower_on=True,
        validation_functions=[],
        use_empty_repair=False,
        run_id_suffix='-mcd-full'
    )
    run_with_specific_features(
        gower_on=False,
        changed_feature_ratio_on=False,
        average_gower_on=False,
        validation_functions=[],
        use_empty_repair=True,
        run_id_suffix='-all-off'
    )
