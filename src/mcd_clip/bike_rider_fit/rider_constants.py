import numpy as np

SAMPLE_RIDER_DICT = {'height': 1869.4399999999998, 'sh_height': 1522.4183722286996, 'hip_to_ankle': 859.4115496065015,
                     'hip_to_knee': 419.2707983114694, 'shoulder_to_wrist': 520.3842323834416,
                     'arm_length': 595.1618323834416, 'torso_length': 588.2292226221981,
                     'lower_leg': 514.9183512950322, 'upper_leg': 419.2707983114694,
                     "foot_length": 5.5 * 25.4, "ankle_angle": 100}


def to_body_vector(body: dict) -> np.ndarray:
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


SAMPLE_RIDER = to_body_vector(SAMPLE_RIDER_DICT)
