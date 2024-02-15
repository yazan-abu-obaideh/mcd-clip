from typing import List

import pandas as pd
from pymoo.core.variable import Variable, Choice, Real

from mcd_clip.biked.load_data import load_augmented_framed_dataset
from mcd_clip.clips_dataset_utils.datatypes_mapper import map_column
from mcd_clip.combined_optimization.columns_constants import FRAMED_COLUMNS, CLIPS_COLUMNS, CLIPS_IGNORED_MATERIAL, \
    FRAMED_TO_CLIPS_IDENTICAL, FRAMED_TO_CLIPS_UNITS, ERGONOMICS_COLUMNS
from mcd_clip.resource_utils import resource_path


def unscaled_framed_dataset():
    x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
    x_columns = x.columns
    x_index = x.index
    return pd.DataFrame(x_scaler.inverse_transform(x), columns=x_columns, index=x_index)


UNSCALED_FRAMED = unscaled_framed_dataset()


def _map_framed_column(result, column):
    name_lower = str(column).lower()
    if ("material" in name_lower) or ("include" in name_lower):
        print(f"Mapped {column} to Choice")
        result.append(Choice(options=(0, 1)))
    else:
        bounds = (UNSCALED_FRAMED[column].quantile(0.01), UNSCALED_FRAMED[column].quantile(0.99))
        print(f"Mapped {column} to Real with bounds {bounds}")
        result.append(Real(bounds=bounds))


def map_combined_datatypes(dataframe: pd.DataFrame) -> List[Variable]:
    result = []
    for column in dataframe.columns:
        if column in FRAMED_COLUMNS:
            _map_framed_column(result, column)
        else:
            mapped_datatype = map_column(dataframe[column])
            print(f"Mapped {column} to {mapped_datatype}")
            result.append(mapped_datatype)
    return result


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

    def get_combined(self):
        return self._data.copy(deep=True)

    def get_as_framed(self) -> pd.DataFrame:
        dropped = self._data.drop(columns=[c for c in self._data.columns if c not in FRAMED_COLUMNS])
        return pd.DataFrame(dropped, columns=FRAMED_COLUMNS)

    def get_for_ergonomics(self) -> pd.DataFrame:
        data = self._data.copy(deep=True)
        data.drop(columns=[c for c in data.columns if c not in ERGONOMICS_COLUMNS], inplace=True)
        data['Stem length'] = 123
        data['Stem angle'] = 100
        data['Handlebar style'] = 2
        data['Crank length'] = 170
        data['Headset spacers'] = 15
        return data

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
    def _replace_intersection(framed: pd.DataFrame, result: pd.DataFrame):
        result['Stack'] = framed['Stack']
        result['DT Length'] = framed['DT Length']

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


class OriginalCombinedDataset:
    def __init__(self):
        x_framed = self._get_framed_unscaled()
        clips = self._get_clips_corrected_index()
        intersection = self._index_intersection(clips, x_framed)
        self._combined = CombinedDataset.build_from_both(framed_style=x_framed.loc[intersection],
                                                         clips_style=clips.loc[intersection])

    def _index_intersection(self, clips, x_framed):
        return list(set(x_framed.index).intersection(set(clips.index)))

    def _get_clips_corrected_index(self):
        clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        clips.index = [str(idx) for idx in clips.index]
        return clips

    def _get_framed_unscaled(self):
        x_framed, y_framed, x_scaler, y_scaler = load_augmented_framed_dataset()
        return pd.DataFrame(x_scaler.inverse_transform(x_framed), columns=x_framed.columns, index=x_framed.index)

    def get_combined_dataset(self) -> CombinedDataset:
        return self._combined
