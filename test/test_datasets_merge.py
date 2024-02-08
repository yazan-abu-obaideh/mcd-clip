import unittest

import pandas as pd

from mcd_clip.resource_utils import resource_path


class DatasetMergeTest(unittest.TestCase):
    def setUp(self):
        self.biked = pd.read_csv(resource_path('all_structural_data_aug.csv'), index_col=0)
        self.clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)

    def test_dataset_len(self):
        self.assertEqual((len(self.biked), len(self.clips)), (14851, 4512))

    def test_intersection_len(self):
        biked_idx_set = set(self.biked.index)
        clips_idx_set = set([str(index) for index in self.clips.index])
        self.assertEqual(len(biked_idx_set.intersection(clips_idx_set)), 4046)

    def test_intersection_columns(self):
        b_columns = set(self.biked.columns)
        c_columns = set(self.clips.columns)
        self.assertEqual(b_columns.intersection(c_columns), {'DT Length', 'Stack'})
