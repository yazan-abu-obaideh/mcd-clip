import unittest
from typing import List

import numpy as np
import numpy.testing as np_test
import pandas as pd

from mcd_clip.structural.load_data import load_augmented_framed_dataset
from mcd_clip.datasets.columns_constants import FRAMED_TO_CLIPS_IDENTICAL, FRAMED_TO_CLIPS_UNITS, \
    CLIPS_COLUMNS, FRAMED_COLUMNS, ERGONOMICS_COLUMNS
from mcd_clip.datasets.combined_datasets import CombinedDataset, map_combined_datatypes, \
    OriginalCombinedDataset
from mcd_clip.resource_utils import resource_path


class DatasetMergeTest(unittest.TestCase):
    def setUp(self):
        self.framed, y, x_scaler, y_scaler = load_augmented_framed_dataset()
        self.framed = pd.DataFrame(x_scaler.inverse_transform(self.framed),
                                   columns=self.framed.columns,
                                   index=self.framed.index)
        self.clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        self.clips.index = [str(idx) for idx in self.clips.index]
        self._index_intersection = self._get_index_intersection()
        self.original_combined = OriginalCombinedDataset().get_combined_dataset()

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

    def test_constants(self):
        self.assertEqual(set(self.framed.columns), set(FRAMED_COLUMNS))
        self.assertEqual(set(self.clips.columns), set(CLIPS_COLUMNS))

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

    def test_fit_columns(self):
        self.assertEqual(
            set(ERGONOMICS_COLUMNS),
            set(self.original_combined.get_for_ergonomics().columns)
        )

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

    @unittest.skip
    def test_columns_with_identical_values(self):
        b_columns = set(self.framed.columns)
        c_columns = set(self.clips.columns)
        framed_to_clips_identical = {}
        framed_to_clips_units = {}
        for b_column in b_columns:
            for c_column in c_columns:
                if self._all_close(framed_column=b_column, clips_column=c_column, multiplier=1):
                    print(f"{b_column} has very close values to {c_column}")
                    framed_to_clips_identical[b_column] = c_column
                elif self._all_close(framed_column=b_column, clips_column=c_column, multiplier=1000):
                    framed_to_clips_units[b_column] = c_column
                    print(f"{b_column} when scaled has very close values to {c_column}")

        self.assertEqual(framed_to_clips_units, FRAMED_TO_CLIPS_UNITS)
        self.assertEqual(framed_to_clips_identical, FRAMED_TO_CLIPS_IDENTICAL)

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

    def _all_close(self, framed_column: str,
                   clips_column: str,
                   multiplier: int):
        return np.isclose(
            self.framed[framed_column].loc[self._index_intersection].values * multiplier,
            self.clips[clips_column].loc[self._index_intersection].values,
            atol=1e-2
        ).all()
