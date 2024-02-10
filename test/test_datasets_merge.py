import unittest
from typing import List

import numpy as np
import numpy.testing as np_test
import pandas as pd

from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.combined_optimization.columns_constants import FRAMED_TO_CLIPS_IDENTICAL, FRAMED_TO_CLIPS_UNITS, \
    CLIPS_COLUMNS
from mcd_clip.combined_optimization.combined_datasets import CombinedDataset, map_combined_datatypes
from mcd_clip.resource_utils import resource_path


class DatasetMergeTest(unittest.TestCase):
    def setUp(self):
        self.framed, y, x_scaler, y_scaler = load_augmented_framed_dataset()
        self.framed = pd.DataFrame(x_scaler.inverse_transform(self.framed),
                                   columns=self.framed.columns,
                                   index=self.framed.index)
        self.clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        self.clips.index = [str(idx) for idx in self.clips.index]

    def test_map_columns(self):
        framed = self.framed.loc[self._get_index_intersection()]
        clips = self.clips.loc[self._get_index_intersection()]
        combined_dataset = CombinedDataset.build_from_both(framed_style=framed, clips_style=clips)
        combined_datatypes = map_combined_datatypes(combined_dataset.get_combined())

    def test_combined_dataset_framed_unchanged(self):
        framed = self.framed.loc[self._get_index_intersection()]
        clips = self.clips.loc[self._get_index_intersection()]
        combined_dataset = CombinedDataset.build_from_both(framed_style=framed, clips_style=clips)

        self.assertEqual(set(combined_dataset.get_as_framed().columns), set(framed.columns))
        self.assertEqual(set(combined_dataset.get_as_clips().columns), set(clips.columns))
        np_test.assert_array_equal(combined_dataset.get_as_framed(), framed)

    def test_combined_dataset_clips_ignore_material(self):
        """Not ideal, but clips changes a bit through the combination: we ignore materials other than
        steel, titanium, or aluminum, and we do some multiplications to handle units which causes some
        floating point errors"""
        framed = self.framed.loc[self._get_index_intersection()]
        clips = self.clips.loc[self._get_index_intersection()]
        combined_dataset = CombinedDataset.build_from_both(framed_style=framed, clips_style=clips)

        material_columns = [c for c in CLIPS_COLUMNS if 'MATERIAL' in c]
        self.assertEqual(6, len(material_columns))

        different_units = list(FRAMED_TO_CLIPS_UNITS.values())
        np_test.assert_allclose(
            combined_dataset.get_as_clips().drop(columns=different_units + material_columns),
            clips.drop(columns=different_units + material_columns),
            rtol=1e-5
        )
        # TODO: investigate this
        np_test.assert_allclose(
            combined_dataset.get_as_clips().drop(columns=material_columns),
            clips.drop(columns=material_columns),
            atol=1e-2
        )

    def test_framed_columns(self):
        print(self.framed.columns)

    def test_clips_columns(self):
        print(self.clips.columns)

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
        framed_subset = self.framed.loc[indices, :]
        clips_subset = self.clips.loc[indices, :]

        np_test.assert_array_almost_equal(
            framed_subset[['DT Length', 'Stack']].values * 1000,
            clips_subset[['DT Length', 'Stack']].values,
            decimal=2
        )

    def test_columns_with_identical_values(self):
        b_columns = set(self.framed.columns)
        c_columns = set(self.clips.columns)
        indices = self._get_index_intersection()
        identical_framed = []
        identical_clips = []
        scaled_framed = []
        scaled_clips = []
        for b_column in b_columns:
            for c_column in c_columns:
                try:
                    if (np.isclose(
                            self.framed[b_column].loc[indices].values,
                            self.clips[c_column].loc[indices].values,
                            atol=1e-2
                    ).all()):
                        print(f"{b_column} has very close values to {c_column}")
                        identical_framed.append(b_column)
                        identical_clips.append(c_column)
                    elif np.isclose(
                            self.framed[b_column].loc[indices].values * 1000,
                            self.clips[c_column].loc[indices].values,
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

        self.assertEqual(len(scaled_framed), len(FRAMED_TO_CLIPS_UNITS))
        self.assertEqual(len(identical_framed), len(FRAMED_TO_CLIPS_IDENTICAL))

    def test_drop_material(self):
        intersection = self._get_index_intersection()
        clips = self.clips.loc[intersection]
        for material in ['MATERIAL OHCLASS: BAMBOO', 'MATERIAL OHCLASS: CARBON', 'MATERIAL OHCLASS: OTHER']:
            print(np.sum(clips[material]))

    def test_get_material(self):
        print([c for c in self.clips.columns if 'material' in str(c).lower()])

    def _get_index_intersection(self) -> List[str]:
        framed_idx_set = set(self.framed.index)
        clips_idx_set = set([str(index) for index in self.clips.index])
        return list(framed_idx_set.intersection(clips_idx_set))
