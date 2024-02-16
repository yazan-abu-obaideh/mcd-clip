import unittest

import numpy.testing as np_test

from mcd_clip.datasets.columns_constants import FRAMED_COLUMNS, CLIPS_COLUMNS, BIKE_FIT_COLUMNS, \
    FRAMED_TO_CLIPS_UNITS, FRAMED_TO_CLIPS_IDENTICAL, UNIQUE_BIKE_FIT_COLUMNS
from mcd_clip.datasets.combined_datasets import OriginalCombinedDataset


class CombinedDatasetTest(unittest.TestCase):
    def setUp(self):
        original_combined_dataset = OriginalCombinedDataset()
        self.original_combined = original_combined_dataset.get_combined_dataset()
        self.clips = original_combined_dataset._get_clips_corrected_index()
        self.framed = original_combined_dataset._get_framed_unscaled()
        self._bike_fit = original_combined_dataset._get_bike_fit_corrected_index()

    def test_intersection_rows(self):
        number_intersection_rows = len(self.original_combined.get_combined())
        self.assertEqual(
            3441,
            number_intersection_rows,
        )

    @unittest.skip
    def test_clips_preservation(self):
        pass

    @unittest.skip
    def test_bike_fit_preservation(self):
        combined_as_fit = self.original_combined.get_as_bike_fit()
        bike_fit = self._bike_fit.loc[combined_as_fit.index]
        self.assertEqual(len(bike_fit), len(combined_as_fit))
        self.assertEqual(len(combined_as_fit), 3441)
        for c in UNIQUE_BIKE_FIT_COLUMNS:
            np_test.assert_array_equal(
                combined_as_fit[c].values,
                bike_fit[c].values
            )

        for c in BIKE_FIT_COLUMNS:
            if c in FRAMED_TO_CLIPS_IDENTICAL.values():
                print(f"Testing identical column {c}...")
                np_test.assert_array_equal(
                    combined_as_fit[c].values,
                    bike_fit[c].values,
                )

        for c in BIKE_FIT_COLUMNS:
            if c in FRAMED_TO_CLIPS_UNITS.values():
                print(f"Testing unit column {c}...")
                np_test.assert_array_equal(
                    combined_as_fit[c].values,
                    bike_fit[c].values,
                )

    def test_column_constants(self):
        self.assertEqual(list(self.framed.columns), FRAMED_COLUMNS)
        self.assertEqual(list(self.clips.columns), CLIPS_COLUMNS)
        self.assertEqual(list(self._bike_fit), BIKE_FIT_COLUMNS)

    def test_column_mapping_constants(self):
        for k, v in FRAMED_TO_CLIPS_UNITS.items():
            self.assertIn(k, FRAMED_COLUMNS)
            self.assertIn(v, CLIPS_COLUMNS)

        for k, v in FRAMED_TO_CLIPS_IDENTICAL.items():
            self.assertIn(k, FRAMED_COLUMNS)
            self.assertIn(v, CLIPS_COLUMNS)

        self.assertOneToOne(FRAMED_TO_CLIPS_UNITS)
        self.assertOneToOne(FRAMED_TO_CLIPS_IDENTICAL)

    def test_dataset_len(self):
        self.assertEqual(len(self.framed), 14851)
        self.assertEqual(len(self.clips), 4512)
        self.assertEqual(len(self._bike_fit), 4046)

    def test_framed_preserved(self):
        combined = self.original_combined.get_combined()
        framed = self.framed.loc[combined.index]
        self.assertEqual(len(framed), len(combined))
        self.assertEqual(len(combined), 3441)
        for c in FRAMED_COLUMNS:
            np_test.assert_array_equal(
                combined[c].values,
                framed[c].values
            )

    def assertOneToOne(self, any_map: dict):
        self.assertEqual(len(any_map.keys()), len(set(any_map.values())))
