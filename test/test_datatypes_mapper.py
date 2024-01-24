import unittest

import pandas as pd
from pymoo.core.variable import Real, Integer, Choice

from datatypes_mapper import map_column


class MapperTest(unittest.TestCase):
    def test_float_mapping(self):
        series = pd.Series()
        series.name = "Head angle"
        series[0] = 15
        series[1] = 25
        column_type = map_column(series)
        self.assertTrue(type(column_type) is Real)
        self.assertEqual(15, column_type.bounds[0])
        self.assertEqual(25, column_type.bounds[1])

    def test_integer_mapping(self):
        series = pd.Series()
        series.name = "Number of chainrings"
        series[0] = 15
        series[1] = 25
        column_type = map_column(series)
        self.assertTrue(type(column_type) is Integer)
        self.assertEqual(15, column_type.bounds[0])
        self.assertEqual(25, column_type.bounds[1])

    def test_bool_mapping(self):
        series = pd.Series()
        series.name = "CHAINSTAYbrdgCheck"
        series[0] = False
        series[1] = True
        series = series.astype(dtype="bool")
        mapped_type = map_column(series)
        self.assertTrue(type(mapped_type) is Choice)
