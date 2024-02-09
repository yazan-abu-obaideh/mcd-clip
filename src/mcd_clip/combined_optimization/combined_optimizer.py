from typing import Set

import pandas as pd

from mcd_clip.resource_utils import resource_path



class CombinedDataset:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def get_as_framed(self) -> pd.DataFrame:
        return

    def get_as_clips(self) -> pd.DataFrame:
        return


class CombinedOptimizer:
    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        pass
