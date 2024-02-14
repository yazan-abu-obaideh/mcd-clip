import unittest

import pandas as pd

from mcd_clip.resource_utils import resource_path


class FitAnalysisTest(unittest.TestCase):

    def setUp(self):
        pd.read_csv(resource_path(''))
