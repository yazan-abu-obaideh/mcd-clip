import pandas as pd

from mcd_clip.combined_optimization.columns_constants import CLIPS_IGNORED_MATERIAL, FRAMED_TO_CLIPS_IDENTICAL, \
    FRAMED_TO_CLIPS_UNITS


class CombinedDataset:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    @classmethod
    def build_from_both(cls, framed_style: pd.DataFrame, clips_style: pd.DataFrame):
        framed = framed_style.copy(deep=True)
        clips = clips_style.copy(deep=True)
        assert len(framed) == len(clips), "Must have the same number of rows to combine"
        CombinedDataset._drop_intersection(clips)
        result = pd.concat([framed, clips], axis=1)
        assert len(result) == len(framed)
        CombinedDataset._drop_clips_material(result)
        CombinedDataset._drop_clips_identical(result)
        CombinedDataset._drop_clips_meter_columns(result)
        return CombinedDataset(result)

    def get_as_framed(self) -> pd.DataFrame:
        data = self._data.copy(deep=True)
        return data.drop(
            columns=[]
        )

    def get_as_clips(self) -> pd.DataFrame:
        data = self._data.copy(deep=True)
        for material in CLIPS_IGNORED_MATERIAL:
            data[material] = 0

        return data

    @staticmethod
    def _drop_clips_material(dataframe: pd.DataFrame):
        clips_material_columns = [c for c in dataframe.columns if 'MATERIAL' in str(c) and 'OHCLASS' in str(c)]
        print(f"Dropping clips material columns {clips_material_columns}")
        dataframe.drop(columns=clips_material_columns, inplace=True)

    @staticmethod
    def _drop_intersection(clips: pd.DataFrame):
        clips.drop(columns=CombinedDataset.get_intersection_columns(), inplace=True)

    @staticmethod
    def get_intersection_columns():
        return ['DT Length', 'Stack']

    @staticmethod
    def _drop_clips_identical(dataframe: pd.DataFrame):
        clips_identical = list(FRAMED_TO_CLIPS_IDENTICAL.values())
        dataframe.drop(columns=clips_identical, inplace=True)

    @staticmethod
    def _drop_clips_meter_columns(dataframe: pd.DataFrame):
        clips_meter_columns = list(FRAMED_TO_CLIPS_UNITS.values())
        dataframe.drop(columns=clips_meter_columns, inplace=True)


class CombinedOptimizer:
    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        pass
