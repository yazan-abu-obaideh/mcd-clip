import unittest

import numpy as np
from sklearn.metrics import r2_score

from mcd_clip.structural.load_data import load_augmented_framed_dataset
from mcd_clip.structural.structural_predictor import StructuralPredictor


class BikedPredictionsTest(unittest.TestCase):
    def setUp(self):
        self.predictor = StructuralPredictor()

    def test_r2_subset(self):
        x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
        x = x.sample(500, random_state=25)
        y = y.loc[x.index]
        score = r2_score(y, self.predictor.predict(x))
        self.assertGreater(score, 0.86)

    @unittest.skip
    def test_r2_full_df(self):
        x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
        numeric_x = x.iloc[:, 5:]
        mask = np.logical_and(np.greater(numeric_x, numeric_x.quantile(0.001)),
                              np.less(numeric_x, numeric_x.quantile(0.999))
                              ).all(axis=1)
        x_no_outliers = x[mask]
        removed_outliers = len(x) - len(x_no_outliers)
        self.assertEqual(920, removed_outliers)
        self.assertGreater(r2_score(y[mask], self.predictor.predict(x_no_outliers)),
                           0.74)
