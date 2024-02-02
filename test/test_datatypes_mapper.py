import unittest

import pandas as pd
from pymoo.core.variable import Real, Integer, Choice

from mcd_clip.clips_dataset_utils.datatypes_mapper import map_column


class MapperTest(unittest.TestCase):
    def test_float_mapping(self):
        series = pd.Series(data=[15, 25], dtype="float64")
        series.name = "Head angle"
        column_type = map_column(series)
        self.assertTrue(type(column_type) is Real)
        self.assertPercentileBounds(column_type)

    def test_integer_mapping(self):
        series = pd.Series(data=[15, 25], dtype="int64")
        series.name = "Number of chainrings"
        column_type = map_column(series)
        self.assertTrue(type(column_type) is Integer)
        self.assertPercentileBounds(column_type)

    def test_bool_mapping(self):
        series = pd.Series(data=[False, True], dtype="bool")
        series.name = "CHAINSTAYbrdgCheck"
        series = series.astype(dtype="bool")
        mapped_type = map_column(series)
        self.assertTrue(type(mapped_type) is Choice)

    def assertPercentileBounds(self, column_type):
        self.assertEqual(15.1, column_type.bounds[0])
        self.assertEqual(24.9, column_type.bounds[1])
