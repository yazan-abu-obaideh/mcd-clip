import numpy as np
import pandas as pd
from decode_mcd import data_package, counterfactuals_generator, design_targets
from decode_mcd import multi_objective_problem as MOP
from pymoo.core.variable import Real


def validity(_x):  # Validity function for the 2D case
    a = _x["X"]  # Separate the two dimensions for clarity
    b = _x["Y"]
    fc = np.less(np.sqrt(np.power((a - 0.3), 2) + np.power((b - 0.3), 2)), 0.1)  # Circle
    sc = np.less(np.power(np.power(np.power((a - b), 6) - 1, 2) + np.power(np.power((a + b), 6) - 1, 2), 2),
                 0.99)  # Arcs
    return np.logical_or(fc, sc)  # If points are in circle or arcs they are valid


all_datapoints = np.random.rand(10000, 2)  # Sample 10000 2D points
all_datapoints = all_datapoints * 2.2 - 1.1  # Scale from -1.1 to 1.1
x_df = pd.DataFrame(all_datapoints, columns=["X", "Y"])
validity_mask = validity(x_df)
y_df = pd.DataFrame(validity_mask, columns=["O1"])
all_df = pd.concat([x_df, y_df], axis=1)
print(all_df)
v = 100 * np.mean(all_df["O1"])
print(f"{v}% of the points are valid")

# importlib.reload(MOCG)
# importlib.reload(calculate_dtai)

data = data_package.DataPackage(features_dataset=x_df,
                                predictions_dataset=y_df,
                                query_x=np.array([[0, 0]]),
                                design_targets=design_targets.DesignTargets([design_targets.ContinuousTarget(label="O1",
                                                                                                             lower_bound=0.9,
                                                                                                             upper_bound=1.1)]),
                                datatypes=[Real(bounds=(-1.1, 1.1)), Real(bounds=(-1.1, 1.1))])

problem = MOP.MultiObjectiveProblem(data_package=data,
                                    prediction_function=validity,
                                    constraint_functions=[])

generator = counterfactuals_generator.CounterfactualsGenerator(problem=problem,
                                                               pop_size=100,
                                                               initialize_from_dataset=True)
generator.generate(n_generations=100)
generator.save("generated_cfs.txt")

weights = generator.sample_with_weights(10, 1, 1, 1, 1)
print(weights)
