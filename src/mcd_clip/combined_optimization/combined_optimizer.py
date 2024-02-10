import pandas as pd

from mcd_clip.combined_optimization.columns_constants import CLIPS_IGNORED_MATERIAL, FRAMED_TO_CLIPS_IDENTICAL, \
    FRAMED_TO_CLIPS_UNITS, FRAMED_COLUMNS, CLIPS_COLUMNS


class CombinedDataset:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    @classmethod
    def build_from_both(cls, framed_style: pd.DataFrame, clips_style: pd.DataFrame):
        framed = framed_style.copy(deep=True)
        clips = clips_style.copy(deep=True)
        assert len(framed) == len(clips), "Must have the same number of rows to combine"
        cls._drop_intersection(clips)
        result = pd.concat([framed, clips], axis=1)
        assert len(result) == len(framed)
        cls._drop_clips_material(result)
        cls._drop_clips_identical(result)
        cls._drop_clips_millimeter_columns(result)
        cls._replace_intersection(framed=framed, result=result)
        return CombinedDataset(result)

    @staticmethod
    def _replace_intersection(framed: pd.DataFrame, result: pd.DataFrame):
        result['Stack'] = framed['Stack']
        result['DT Length'] = framed['DT Length']

    def get_as_framed(self) -> pd.DataFrame:
        dropped = self._data.drop(columns=[c for c in self._data.columns if c not in FRAMED_COLUMNS])
        return pd.DataFrame(dropped, columns=FRAMED_COLUMNS)

    def get_as_clips(self) -> pd.DataFrame:
        data = self._data.copy(deep=True)
        self._to_clips_material(data)
        self._to_millimeter_columns(data)
        self._to_clips_identical(data)
        return pd.DataFrame(data, columns=CLIPS_COLUMNS)

    def _to_clips_material(self, data: pd.DataFrame):
        for material in CLIPS_IGNORED_MATERIAL:
            data[material] = 0
        framed_material_columns = [c for c in data.columns if 'Material' in c]
        print(f"Found framed material columns {framed_material_columns}")
        for material_column in framed_material_columns:
            clips_column = self._to_clips_material_column(material_column)
            data[clips_column] = data[material_column]
        data.drop(columns=framed_material_columns, inplace=True)
        self._handle_aluminum(data)

    def _to_clips_material_column(self, material_column: str) -> str:
        material_value = material_column.split("=")[1]
        clips_column = f"MATERIAL OHCLASS: {material_value.upper()}"
        print(f"Mapping {material_column} to {clips_column}")
        return clips_column

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
    def _drop_clips_millimeter_columns(dataframe: pd.DataFrame):
        clips_meter_columns = list(FRAMED_TO_CLIPS_UNITS.values())
        dataframe.drop(columns=clips_meter_columns, inplace=True)

    def _to_millimeter_columns(self, data: pd.DataFrame):
        data.drop(columns=FRAMED_TO_CLIPS_UNITS.keys(), inplace=True)
        for framed_key, clips_key in FRAMED_TO_CLIPS_UNITS.items():
            data[clips_key] = self._data[framed_key] * 1000

    def _to_clips_identical(self, data: pd.DataFrame):
        data.drop(columns=FRAMED_TO_CLIPS_IDENTICAL.keys(), inplace=True)
        for framed_key, clips_key in FRAMED_TO_CLIPS_IDENTICAL.items():
            data[clips_key] = self._data[framed_key]

    def _handle_aluminum(self, data):
        american_spelling = 'MATERIAL OHCLASS: ALUMINUM'
        data['MATERIAL OHCLASS: ALUMINIUM'] = data[american_spelling]
        data.drop(columns=[american_spelling], inplace=True)


class CombinedOptimizer:
    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        pass
