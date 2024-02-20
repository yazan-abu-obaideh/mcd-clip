import numpy as np
import matplotlib.pyplot as plt


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

        x = np.random.rand(1000, 2) * 5 - 5
        y = np.random.rand(1000, 2) * 5 - 5
        # Plot the points
        ax.scatter(x, y, color='black', s=5)

        # Set the aspect of the plot to be equal, so the circles appear as circles
        ax.set_aspect('equal')

        # Set the limits of the plot's x and y axes
        ax.set_xlim(-max(radii) - 1, max(radii) + 1)
        ax.set_ylim(-max(radii) - 1, max(radii) + 1)

        plt.savefig('circles.png')


if __name__ == '__main__':
    ablation_study = SimpleAblation2Dimensions()
    ablation_study.run_case_study()
