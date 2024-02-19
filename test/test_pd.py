import unittest

import numpy as np
import pandas as pd


class PandasWeirdBehavior(unittest.TestCase):
    def test_cast_index(self):
        result = np.ones(shape=(100,))
        result = pd.DataFrame(result)
        print(result)
        result = pd.DataFrame(result, index=[i for i in range(50, 150)])
        print(result)
