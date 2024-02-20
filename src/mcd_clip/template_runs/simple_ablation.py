import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decode_mcd import data_package, MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget, DesignTargets, \
    DataPackage
from pymoo.core.variable import Real


class SimpleAblation2Dimensions:
    def run_case_study(self):
        self._draw_valid_region()
        self._run_without_features()
        self._run_with_features()

    def _run_without_features(self):
        pass

    def _run_with_features(self):
        pass

    def _run(self, features: bool):
        pass

    def validity(self, _x):  # Validity function for the 2D case
        a = _x["X"]  # Separate the two dimensions for clarity
        b = _x["Y"]
        fc = np.less(np.sqrt(np.power((a - 0.3), 2) + np.power((b - 0.3), 2)), 0.1)  # Circle
        sc = np.less(np.power(np.power(np.power((a - b), 6) - 1, 2) + np.power(np.power((a + b), 6) - 1, 2), 2),
                     0.99)  # Arcs
        return np.logical_or(fc, sc)  # If points are in circle or arcs they are valid

    def _draw_arcs(self):
        pass

    def plotcfs(self, validity, counterfactuals, query, rangearr, dataset=False):
        xx, yy = np.mgrid[rangearr[0, 0]:rangearr[0, 1]:.001, rangearr[1, 0]:rangearr[1, 1]:.001]
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_df = pd.DataFrame(grid, columns=["X", "Y"])
        Z = validity(grid_df)
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        img = ax.imshow(1 - Z.T, cmap="gray", alpha=0.5, origin='lower', extent=rangearr.flatten())
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

    def _draw_valid_region(self):
        # Define the radii of the circles
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

        self._do_2d_case()

        plt.savefig('circles.png')

    def plotcfs(self,
                validity,
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
        img = ax.imshow(1 - Z.T, cmap="gray", alpha=0.5, origin='lower', extent=rangearr.flatten())
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

    def _do_2d_case(self):
        all_datapoints = np.random.rand(10000, 2)  # Sample 10000 2D points
        all_datapoints = all_datapoints * 2.2 - 1.1  # Scale from -1.1 to 1.1
        x_df = pd.DataFrame(all_datapoints, columns=["X", "Y"])
        validity_mask = self.validity(x_df)
        y_df = pd.DataFrame(validity_mask, columns=["O1"])
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

        generator = CounterfactualsGenerator(problem=problem,
                                             pop_size=100,
                                             initialize_from_dataset=True)
        generator.generate(n_generations=100)
        counterfactuals = generator.sample_with_dtai(num_samples=10, gower_weight=1,
                                                     avg_gower_weight=0.5, cfc_weight=0.5,
                                                     diversity_weight=0.2, include_dataset=False,
                                                     num_dpp=10000)
        self.plotcfs(self.validity,
                     counterfactuals.values,
                     (0, 0),
                     np.array([[-1.1, 1.1], [-1.1, 1.1]]))


if __name__ == '__main__':
    ablation_study = SimpleAblation2Dimensions()
    ablation_study.run_case_study()
