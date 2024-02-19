import unittest

from mcd_clip.datasets.category_maps import CATEGORY_MAPS
from mcd_clip.datasets.combined_datasets import is_categorical, OriginalCombinedDataset


class CategoryMapsTest(unittest.TestCase):
    def test_is_categorical(self):
        category_map = {}
        original_combined = OriginalCombinedDataset().get_combined_dataset()
        for c in original_combined.get_combined().columns:
            if is_categorical(c):
                unique_values = list(original_combined.get_combined()[c].unique())
                unique_values.sort()
                category_map[c] = {
                    i: unique_values[i] for i in range(len(unique_values))
                }
        self.assertEqual(CATEGORY_MAPS,
                         category_map
                         )
