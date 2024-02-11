import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcd_clip.biked.MultilabelPredictor import MultilabelPredictor
from mcd_clip.resource_utils import resource_path
import __main__

_MODEL_PATH = resource_path('generated/AutogluonModels/ag-20231016_092811')


class StructuralPredictor:
    def __init__(self):
        __main__.MultilabelPredictor = MultilabelPredictor
        self._predictor = MultilabelPredictor.load(_MODEL_PATH)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._predictor.predict(x)

    def predict_unscaled(self, x: pd.DataFrame, x_scaler: StandardScaler, y_scaler: StandardScaler):
        x_scaled = x_scaler.transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)
        scaled_predictions = self.predict(x_scaled)
        unscaled_predictions = y_scaler.inverse_transform(scaled_predictions)
        return pd.DataFrame(unscaled_predictions, columns=scaled_predictions.columns, index=scaled_predictions.index)
