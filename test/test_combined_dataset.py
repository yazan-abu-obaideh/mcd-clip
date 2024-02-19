import unittest

import numpy.testing as np_test

from mcd_clip.datasets.columns_constants import FRAMED_COLUMNS, CLIPS_COLUMNS, BIKE_FIT_COLUMNS, \
    FRAMED_TO_CLIPS_UNITS, FRAMED_TO_CLIPS_IDENTICAL, UNIQUE_BIKE_FIT_COLUMNS, FRAMED_CLIPS_INTERSECTION_COLUMNS, \
    ONE_HOT_ENCODED_CLIPS_COLUMNS, CLIPS_ONE_HOT_ENCODING_SEP
from mcd_clip.datasets.combined_datasets import OriginalCombinedDataset, CombinedDataset


class CombinedDatasetTest(unittest.TestCase):
    def setUp(self):
        original_combined_dataset = OriginalCombinedDataset()
        self.original_combined = original_combined_dataset.get_combined_dataset()
        self.clips = original_combined_dataset._get_clips_corrected_index()
        self.framed = original_combined_dataset._get_framed_unscaled()
        self._bike_fit = original_combined_dataset._get_bike_fit_corrected_index()

    def test_constructor(self):
        dataset = CombinedDataset(self.original_combined.get_combined())
        np_test.assert_array_equal(
            dataset.get_as_framed(),
            self.original_combined.get_as_framed()
        )
        np_test.assert_array_equal(
            dataset.get_as_clips(),
            self.original_combined.get_as_clips()
        )
        np_test.assert_array_equal(
            dataset.get_as_bike_fit(),
            self.original_combined.get_as_bike_fit()
        )

    def test_intersection_rows(self):
        number_intersection_rows = len(self.original_combined.get_combined())
        self.assertEqual(
            4046,
            number_intersection_rows,
        )

    def test_clips_preservation(self):
        combined_as_clips = self.original_combined.get_as_clips()
        clips = self.clips.loc[combined_as_clips.index]
        self.assertEqual(len(clips), len(combined_as_clips))
        self.assertEqual(len(combined_as_clips), 4046)
        relevant_clips_columns = [c for c in CLIPS_COLUMNS if
                                  ('material' not in c.lower()
                                   and c not in FRAMED_TO_CLIPS_UNITS.values()
                                   and c not in FRAMED_TO_CLIPS_IDENTICAL.values()
                                   )
                                  ]
        self.assertEqual(len(relevant_clips_columns), 64)
        for clips_column in relevant_clips_columns:
            np_test.assert_array_equal(
                combined_as_clips[clips_column],
                clips[clips_column],
            )

    def test_bike_fit_preservation(self):
        combined_as_fit = self.original_combined.get_as_bike_fit()
        bike_fit = self._bike_fit.loc[combined_as_fit.index]
        self.assertEqual(len(bike_fit), len(combined_as_fit))
        self.assertEqual(len(combined_as_fit), 4046)
        np_test.assert_allclose(
            bike_fit, combined_as_fit,
            rtol=1e-05
        )
        for c in UNIQUE_BIKE_FIT_COLUMNS:
            np_test.assert_array_equal(
                combined_as_fit[c].values,
                bike_fit[c].values
            )

    def test_column_constants(self):
        self.assertEqual(list(self.framed.columns), FRAMED_COLUMNS)
        self.assertEqual(list(self.clips.columns), CLIPS_COLUMNS)
        self.assertEqual(list(self._bike_fit), BIKE_FIT_COLUMNS)

        self.assertEqual(set(ONE_HOT_ENCODED_CLIPS_COLUMNS),
                         set([str(column).split(CLIPS_ONE_HOT_ENCODING_SEP)[0]
                              for column in self.clips.columns if 'OHCLASS' in column]))
        self.assertEqual(set(FRAMED_CLIPS_INTERSECTION_COLUMNS),
                         set(self.framed.columns).intersection(self.clips.columns)
                         )

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
        combined = self.original_combined.get_as_framed()
        framed = self.framed.loc[combined.index]
        self.assertEqual(len(framed), len(combined))
        self.assertEqual(len(combined), 4046)
        for c in FRAMED_COLUMNS:
            if 'Material' in c:
                np_test.assert_array_equal(
                    combined[c].values.astype('int32'),
                    framed[c].values.astype('int32')
                )
            else:
                np_test.assert_array_equal(
                    combined[c].values,
                    framed[c].values
                )

    def assertOneToOne(self, any_map: dict):
        self.assertEqual(len(any_map.keys()), len(set(any_map.values())))
