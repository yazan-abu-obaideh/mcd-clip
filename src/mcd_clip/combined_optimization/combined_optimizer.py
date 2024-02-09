from typing import Set

import pandas as pd

from mcd_clip.resource_utils import resource_path


class CombinedDataset:
    def __init__(self):
        self._framed = pd.read_csv(resource_path('all_structural_data_aug.csv'), index_col=0)
        self._clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        self._clips.index = [str(idx) for idx in self._clips.index]

    def get_shared_index(self) -> Set[str]:
        return set(self._framed.index).intersection(self._clips.index)

    def get_original_framed(self) -> pd.DataFrame:
        return self._framed

    def get_original_clips(self) -> pd.DataFrame:
        return self._clips


class CombinedOptimizer:
    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        pass
