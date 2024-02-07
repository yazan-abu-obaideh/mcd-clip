import unittest

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.biked.structural_predictor import StructuralPredictor


class BikedPredictionsTest(unittest.TestCase):
    def setUp(self):
        self.predictor = StructuralPredictor()
        self.x, self.y, self.x_scaler, self.y_scaler = self.prepare_x_y()

    def test_r2(self):
        predictions = self.predictor.predict(self.x)
        score = r2_score(self.y, predictions)
        self.assertGreater(score, 0.73)

    def prepare_x_y(self):
        x_scaled, y_scaled, x_scaler, y_scaler = load_augmented_framed_dataset()
        x_test, y_test = self.standard_split(x_scaled, y_scaled)
        return x_test, y_test, x_scaler, y_scaler

    def standard_split(self, x_scaled, y_scaled):
        x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                                            y_scaled,
                                                            test_size=0.2,
                                                            random_state=1950)
        return x_test, y_test
