import unittest

import pandas as pd

from mcd_clip.bike_embedding.ordered_columns import ORDERED_COLUMNS
from mcd_clip.resource_utils import resource_path


class OrderedColumnsTest(unittest.TestCase):
    def setUp(self):
        self.original_data = pd.read_csv(resource_path("clip_sBIKED_processed.csv"), index_col=0)

    def test_matches_dataset(self):
        self.assertEqual(
            ORDERED_COLUMNS,
            list(self.original_data.columns)
        )
