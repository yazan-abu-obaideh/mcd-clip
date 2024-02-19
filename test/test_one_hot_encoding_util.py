import unittest

import numpy as np
import pandas as pd

from mcd_clip.datasets.one_hot_encoding_util import get_encoded_columns, reverse_one_hot_encoding
from mcd_clip.resource_utils import resource_path


class OneHotEncodingUtilTest(unittest.TestCase):
    def setUp(self):
        self.framed_original = pd.read_csv(resource_path('all_structural_data_aug.csv'), index_col=0)
        self.clips_original = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)

    def test_framed_encode(self):
        encoded_materials = get_encoded_columns(self.framed_original,
                                                column_name='Material',
                                                prefix_sep='='
                                                )
        revered_encoding = reverse_one_hot_encoding(encoded_materials,
                                                    column_finder=lambda column_name: 'material' in column_name.lower(),
                                                    separator='=')
        np.testing.assert_array_equal(
            self.framed_original['Material'].values,
            revered_encoding['Material'].values
        )

    def test_clips_encode(self):
        encoding = reverse_one_hot_encoding(self.clips_original,
                                            column_finder=lambda column_name: 'OHCLASS' in column_name.upper(),
                                            separator=' OHCLASS: ')
        print(encoding)
