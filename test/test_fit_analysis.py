import unittest

import numpy as np
import pandas as pd

from mcd_clip.bike_rider_fit.fit_analysis.demoanalysis_wrapped import calculate_angles, calculate_drag
from mcd_clip.resource_utils import resource_path

SAMPLE_RIDER = {'height': 1869.4399999999998, 'sh_height': 1522.4183722286996, 'hip_to_ankle': 859.4115496065015,
                'hip_to_knee': 419.2707983114694, 'shoulder_to_wrist': 520.3842323834416,
                'arm_length': 595.1618323834416, 'torso_length': 588.2292226221981,
                'lower_leg': 514.9183512950322, 'upper_leg': 419.2707983114694}


class FitAnalysisTest(unittest.TestCase):

    def setUp(self):
        bike_vector_with_id = pd.read_csv(resource_path('bike_vector_df_with_id.csv'), index_col=0)
        self.bike_dataframe = bike_vector_with_id.drop(columns=["Bike ID"])
        self.body_vector = self.to_body_vector(self._to_full_dimensions(SAMPLE_RIDER))

    def test_ergonomics(self):
        calculated_angles = calculate_angles(self.bike_dataframe.values,
                                             self.body_vector)
        for column in ['Knee Extension', 'Back Angle', 'Armpit Angle']:
            self.assertTrue(column in calculated_angles.columns)

    def test_aerodynamics(self):
        predicted_drag = calculate_drag(self.bike_dataframe.values, self.body_vector)
        self.assertTrue("Aerodynamic Drag" in predicted_drag.columns)

    def _to_full_dimensions(self, body: dict):
        body["foot_length"] = 5.5 * 25.4
        body["ankle_angle"] = 100
        return body

    def to_body_vector(self, body: dict):
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
