import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decode_mcd import MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget, DesignTargets, \
    DataPackage
from pymoo.core.variable import Real


class SimpleAblation2Dimensions:
    def run_case_study(self):
        self._run_without_features()
        self._run_with_features()

    def secret_validity(self, points: pd.DataFrame):
        distance_ = np.sum(np.power(points.values, 2), axis=1)
        return np.logical_and(distance_ < 1, distance_ > 0.9)

    def _run_without_features(self):
        self._do_2d_case(ablation=True)

    def _run_with_features(self):
        self._do_2d_case(ablation=False)

    def _run(self, features: bool):
        pass

    def validity(self, _x):  # Validity function for the 2D case
        a = _x["X"]  # Separate the two dimensions for clarity
        b = _x["Y"]
        point_in_arcs = np.less(
            np.power(np.power(np.power((a - b), 6) - 1, 2) + np.power(np.power((a + b), 6) - 1, 2), 2), 0.99)
        return point_in_arcs

    def _draw_arcs(self):
        pass

    def _draw_circles(self):
        radii = [5, 4, 3, 2, 1]

        # Define the colors for the circles
        colors = ['red', 'blue'] * len(radii)

        # Create a figure and an axis
        fig, ax = plt.subplots()

        # For each radius, draw a circle
        for r, color in zip(radii, colors):
            circle = plt.Circle((0, 0), r, color=color, alpha=0.5)
            ax.add_artist(circle)

        # Set the aspect of the plot to be equal, so the circles appear as circles
        ax.set_aspect('equal')

        # Set the limits of the plot's x and y axes
        ax.set_xlim(-max(radii) - 1, max(radii) + 1)
        ax.set_ylim(-max(radii) - 1, max(radii) + 1)

        plt.savefig('circles.png')

    def plot_cfs(self,
                 counterfactuals,
                 query,
                 rangearr,
                 dataset=False):
        xx, yy = np.mgrid[rangearr[0, 0]:rangearr[0, 1]:.001, rangearr[1, 0]:rangearr[1, 1]:.001]
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_df = pd.DataFrame(grid, columns=["X", "Y"])
        Z = self.validity(grid_df)
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        plt.Circle((0, 0), 0.5, color='blue', alpha=0.5)
        ax.imshow(1 - Z.T, cmap="gray", alpha=0.5, origin='lower', extent=rangearr.flatten())
        ax.axis('off')

        ax.scatter(query[0], query[1], s=100, c="k", alpha=1, marker="x")
        ax.annotate("Query", (query[0] - .15, query[0] + 0.1))
        if not dataset:
            ax.scatter(counterfactuals[:, 0], counterfactuals[:, 1], s=100, c="k", alpha=1, marker=".")
            for i in range(np.shape(counterfactuals)[0]):
                ax.plot([query[0], counterfactuals[i, 0]], [query[1], counterfactuals[i, 1]], c="k", linestyle="--",
                        lw=1, alpha=0.5)
        else:
            ax.scatter(counterfactuals[:, 0], counterfactuals[:, 1], s=1, c="k", alpha=1, marker=".")

        circle = plt.Circle((0, 0), 1, color='red', alpha=0.5)
        ax.add_artist(circle)
        circle = plt.Circle((0, 0), 0.95, color='white', alpha=0.5)
        ax.add_artist(circle)

    def _do_2d_case(self, ablation: bool):
        all_datapoints = np.random.rand(10000, 2)  # Sample 10000 2D points
        all_datapoints = all_datapoints * 2.2 - 1.1  # Scale from -1.1 to 1.1
        x_df = pd.DataFrame(all_datapoints, columns=["X", "Y"])
        validity_mask = np.logical_and(self.validity(x_df), self.secret_validity(x_df))
        y_df = pd.DataFrame(validity_mask, columns=["O1"])
        x_df = x_df[validity_mask]
        y_df = y_df[validity_mask]
        all_df = pd.concat([x_df, y_df], axis=1)
        v = 100 * np.mean(all_df["O1"])
        print(f"{v}% of the points are valid")
        data = DataPackage(features_dataset=x_df,
                           predictions_dataset=y_df,
                           query_x=np.array([[0, 0]]),
                           design_targets=DesignTargets(
                               [ContinuousTarget(label="O1",
                                                 lower_bound=0.9,
                                                 upper_bound=1.1)]),
                           datatypes=[Real(bounds=(-1.1, 1.1)), Real(bounds=(-1.1, 1.1))])

        problem = MultiObjectiveProblem(data_package=data,
                                        prediction_function=self.validity,
                                        constraint_functions=[])

        features_desired = not ablation

        problem.set_desired_scores(
            gower=features_desired,
            average_gower=features_desired,
            change_feature_ratio=features_desired
        )

        generator = CounterfactualsGenerator(problem=problem,
                                             pop_size=100,
                                             initialize_from_dataset=True)
        generator.use_empty_repair(ablation)
        generator.generate(n_generations=100)
        counterfactuals = generator.sample_with_dtai(num_samples=50, gower_weight=0.5,
                                                     avg_gower_weight=10, cfc_weight=0.5,
                                                     diversity_weight=0.2, include_dataset=False,
                                                     num_dpp=10000)
        self.plot_cfs(counterfactuals.values,
                      (0, 0),
                      np.array([[-1.1, 1.1], [-1.1, 1.1]]))
        if ablation:
            name = 'figure-classical.png'
        else:
            name = 'figure-mcd.png'
        validity = self.secret_validity(counterfactuals)

        fraction_valid = np.sum(validity) / len(validity)
        print(f'Fraction valid: {fraction_valid}')
        plt.savefig(name)


if __name__ == '__main__':
    ablation_study = SimpleAblation2Dimensions()
    ablation_study.run_case_study()
