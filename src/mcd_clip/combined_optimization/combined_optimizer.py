import pandas as pd
from decode_mcd import DataPackage, DesignTargets, CounterfactualsGenerator, MultiObjectiveProblem, ContinuousTarget

from mcd_clip.bike_embedding.embedding_predictor import EmbeddingPredictor
from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.biked.structural_predictor import StructuralPredictor
from mcd_clip.combined_optimization.combined_datasets import CombinedDataset, map_combined_datatypes
from mcd_clip.resource_utils import resource_path

STRUCTURAL_PREDICTOR = StructuralPredictor()
EMBEDDING_PREDICTOR = EmbeddingPredictor()

framed_x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
framed_x = pd.DataFrame(x_scaler.inverse_transform(framed_x),
                        columns=framed_x.columns,
                        index=framed_x.index)
clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
clips.index = [str(idx) for idx in clips.index]

intersection = set(framed_x.index).intersection(set(clips.index))

framed_x = framed_x.loc[intersection]
clips = clips.loc[intersection]

print(len(framed_x))
print(len(clips))


class CombinedOptimizer:
    def predict(self, designs: CombinedDataset) -> pd.DataFrame:
        predictions = STRUCTURAL_PREDICTOR.predict(designs.get_combined())
        return pd.DataFrame(predictions, columns=['Model Mass'])


if __name__ == '__main__':
    optimizer = CombinedOptimizer()
    original_combined = CombinedDataset.build_from_both(framed_style=framed_x, clips_style=clips)
    data_package = DataPackage(
        features_dataset=original_combined.get_combined(),
        predictions_dataset=optimizer.predict(original_combined),
        query_x=original_combined.get_combined().iloc[0:1],
        design_targets=DesignTargets(
            continuous_targets=[ContinuousTarget('Model Mass', lower_bound=0, upper_bound=2)]
        ),
        datatypes=map_combined_datatypes(original_combined.get_combined())
    )

    problem = MultiObjectiveProblem(
        data_package=data_package,
        prediction_function=lambda d: optimizer.predict(CombinedDataset(
            pd.DataFrame(d, columns=original_combined.get_combined().columns))),
        constraint_functions=[]
    )

    generator = CounterfactualsGenerator(
        problem=problem,
        pop_size=100,
        initialize_from_dataset=True
    )

    generator.generate(
        n_generations=400,
    )
    sampled = generator.sample_with_weights(num_samples=100, avg_gower_weight=1, gower_weight=1, cfc_weight=1,
                                            diversity_weight=1)
    print(sampled)
