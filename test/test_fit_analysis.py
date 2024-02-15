import unittest

import pandas as pd

from mcd_clip.bike_rider_fit.fit_analysis.demoanalysis_wrapped import calculate_angles, calculate_drag
from mcd_clip.bike_rider_fit.rider_constants import SAMPLE_RIDER
from mcd_clip.resource_utils import resource_path


class FitAnalysisTest(unittest.TestCase):

    def setUp(self):
        bike_vector_with_id = pd.read_csv(resource_path('bike_vector_df_with_id.csv'), index_col=0)
        self.bike_dataframe = bike_vector_with_id.drop(columns=["Bike ID"])

    def test_ergonomics(self):
        calculated_angles = calculate_angles(self.bike_dataframe.values,
                                             SAMPLE_RIDER)
        for column in ['Knee Extension', 'Back Angle', 'Armpit Angle']:
            self.assertTrue(column in calculated_angles.columns)

    def test_aerodynamics(self):
        predicted_drag = calculate_drag(self.bike_dataframe.values, SAMPLE_RIDER)
        self.assertTrue("Aerodynamic Drag" in predicted_drag.columns)

    def _to_full_dimensions(self, body: dict):
        body["foot_length"] = 5.5 * 25.4
        body["ankle_angle"] = 100
        return body

