import pandas as pd

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
