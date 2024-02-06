import unittest

import pandas as pd
from mcd_clip.resource_utils import resource_path

from mcd_clip.bike_rendering.parametric_to_image_convertor import ParametricToImageConvertor


class RenderingTest(unittest.TestCase):
    def test_rendering(self):
        data = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        rendering_result = ParametricToImageConvertor().to_image(data.iloc[0])
        self.assertTrue(type(rendering_result.bike_xml) is str)
        self.assertTrue(type(rendering_result.image) is bytes)

    def test_one_hot_encoded(self):
        pass
