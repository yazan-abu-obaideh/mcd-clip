import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcd_clip.structural.MultilabelPredictor import MultilabelPredictor
from mcd_clip.resource_utils import resource_path
import __main__

_MODEL_PATH = resource_path('generated/AutogluonModels/ag-20231016_092811')


class StructuralPredictor:
    def __init__(self, exclude_predictors=None):
        __main__.MultilabelPredictor = MultilabelPredictor
        self._predictor = MultilabelPredictor.load(_MODEL_PATH)
        if exclude_predictors is None:
            exclude_predictors = []
        for excluded in exclude_predictors:
            self._predictor: MultilabelPredictor
            del self._predictor.predictors[excluded]

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._predictor.predict(x)

    def predict_unscaled(self, unscaled_x: pd.DataFrame, x_scaler: StandardScaler, y_scaler: StandardScaler):
        x_scaled = x_scaler.transform(unscaled_x)
        x_scaled = pd.DataFrame(x_scaled, columns=unscaled_x.columns, index=unscaled_x.index)
        scaled_predictions = self.predict(x_scaled)
        unscaled_predictions = y_scaler.inverse_transform(scaled_predictions)
        return pd.DataFrame(unscaled_predictions, columns=scaled_predictions.columns, index=scaled_predictions.index)
