import numpy as np

from mcd_clip.fit_analysis.demoanalysis import bike_body_calculation, ergonomics_bike_body_calculation

INFINITY = float("inf")
DEFAULT_ARM_ANGLE = 150


def to_body_vector(body: dict):
    return np.array([
        [body["lower_leg"],
         body["upper_leg"],
         body["torso_length"],
         body["arm_length"],
         body["foot_length"],
         body["ankle_angle"],
         body["shoulder_to_wrist"],
         body["height"]]
    ])


def calculate_angles(bikes: np.ndarray, body: np.ndarray):
    return ergonomics_bike_body_calculation(bikes, body).fillna(INFINITY)


def calculate_drag(bikes: np.ndarray, body: np.ndarray):
    return bike_body_calculation(bikes, body).drop(
        columns=["Knee Extension", "Back Angle", "Armpit Angle"])
