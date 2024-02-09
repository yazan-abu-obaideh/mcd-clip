import unittest
from typing import List

import numpy as np
import numpy.testing as np_test
import pandas as pd

from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.resource_utils import resource_path


class DatasetMergeTest(unittest.TestCase):
    def setUp(self):
        self.framed, y, x_scaler, y_scaler = load_augmented_framed_dataset()
        self.framed = pd.DataFrame(x_scaler.inverse_transform(self.framed),
                                   columns=self.framed.columns,
                                   index=self.framed.index)
        self.clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)

    def test_outliers(self):
        for column in self.framed.columns:
            print(self.framed[column].describe())

    def test_dataset_len(self):
        self.assertEqual((len(self.framed), len(self.clips)), (14851, 4512))

    def test_intersection_len(self):
        self.assertEqual(len(self._get_index_intersection()), 4046)

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
        identical_framed = []
        identical_clips = []
        scaled_framed = []
        scaled_clips = []
        for b_column in b_columns:
            for c_column in c_columns:
                try:
                    if (np.isclose(
                            self.framed[b_column].loc[indices].values,
                            self.clips[c_column].loc[clip_indices].values,
                            atol=1e-2
                    ).all()):
                        print(f"{b_column} has very close values to {c_column}")
                        identical_framed.append(b_column)
                        identical_clips.append(c_column)
                    elif np.isclose(
                            self.framed[b_column].loc[indices].values * 1000,
                            self.clips[c_column].loc[clip_indices].values,
                            atol=1e-2
                    ).all():
                        scaled_framed.append(b_column)
                        scaled_clips.append(c_column)
                        print(f"{b_column} when scaled has very close values to {c_column}")
                except Exception as e:
                    print(f"Exception occurred because of {b_column} and {c_column}")
                    raise e

        print(f"{scaled_framed=}")
        print(f"{scaled_clips=}")
        print(f"{identical_framed=}")
        print(f"{identical_clips=}")

        self.assertEqual(len(scaled_framed), len(scaled_clips))
        self.assertEqual(len(identical_framed), len(identical_clips))

    def _get_index_intersection(self) -> List[str]:
        framed_idx_set = set(self.framed.index)
        clips_idx_set = set([str(index) for index in self.clips.index])
        return list(framed_idx_set.intersection(clips_idx_set))
