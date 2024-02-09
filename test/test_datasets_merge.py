import unittest
from typing import List

import numpy as np
import numpy.testing as np_test
import pandas as pd

from mcd_clip.combined_optimization.combined_optimizer import CombinedDataset
from mcd_clip.resource_utils import resource_path


class DatasetMergeTest(unittest.TestCase):
    def setUp(self):
        self.framed = pd.read_csv(resource_path('all_structural_data_aug.csv'), index_col=0).drop(columns=['batch'])
        self.clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        self.combined = CombinedDataset()

    def test_outliers(self):
        for column in self.framed.columns:
            print(self.framed[column].describe())

    def test_dataset_len(self):
        len_framed = len(self.combined.get_original_framed())
        len_clips = len(self.combined.get_original_clips())
        self.assertEqual((len_framed, len_clips), (14851, 4512))

    def test_intersection_len(self):
        self.assertEqual(len(self.combined.get_shared_index()), 4046)

    def test_intersection_columns(self):
        b_columns = set(self.framed.columns)
        c_columns = set(self.clips.columns)
        self.assertEqual(b_columns.intersection(c_columns), {'DT Length', 'Stack'})

        indices = self._get_index_intersection()
        clips_indices = [int(idx) for idx in indices]
        framed_subset = self.framed.loc[indices, :]
        clips_subset = self.clips.loc[clips_indices, :]

        np_test.assert_array_almost_equal(
            framed_subset[['DT Length', 'Stack']].values * 1000,
            clips_subset[['DT Length', 'Stack']].values,
            decimal=2
        )

    def test_columns_with_identical_values(self):
        b_columns = set(self.framed.columns)
        c_columns = set(self.clips.columns)
        indices = self._get_index_intersection()
        clip_indices = [int(idx) for idx in indices]
        b_columns = {c for c in b_columns if c not in ['Material']}
        for b_column in b_columns:
            for c_column in c_columns:
                try:
                    if (np.isclose(
                            self.framed[b_column].loc[indices].values,
                            self.clips[c_column].loc[clip_indices].values,
                            atol=1e-2
                    ).all()):
                        print(f"{b_column} has very close values to {c_column}")
                    elif np.isclose(
                            self.framed[b_column].loc[indices].values * 1000,
                            self.clips[c_column].loc[clip_indices].values,
                            atol=1e-2
                    ).all():
                        print(f"{b_column} when scaled has very close values to {c_column}")
                except Exception as e:
                    print(f"Exception occurred because of {b_column} and {c_column}")
                    raise e

    def _get_index_intersection(self) -> List[str]:
        framed_idx_set = set(self.framed.index)
        clips_idx_set = set([str(index) for index in self.clips.index])
        return list(framed_idx_set.intersection(clips_idx_set))
