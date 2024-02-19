import random
from typing import List, Callable

import numpy as np
import pandas as pd
from pymoo.core.variable import Variable, Choice, Real

from mcd_clip.datasets.category_maps import CATEGORY_MAPS
from mcd_clip.datasets.clips.datatypes_mapper import map_column
from mcd_clip.datasets.columns_constants import FRAMED_COLUMNS, CLIPS_COLUMNS, CLIPS_IGNORED_MATERIAL, \
    FRAMED_TO_CLIPS_IDENTICAL, FRAMED_TO_CLIPS_UNITS, BIKE_FIT_COLUMNS, UNIQUE_BIKE_FIT_COLUMNS, \
    FRAMED_CLIPS_INTERSECTION_COLUMNS, ONE_HOT_ENCODED_CLIPS_COLUMNS, CLIPS_ONE_HOT_ENCODING_SEP
from mcd_clip.datasets.one_hot_encoding_util import reverse_one_hot_encoding, get_encoded_columns
from mcd_clip.resource_utils import resource_path
from mcd_clip.structural.load_data import load_augmented_framed_dataset

BIKE_FIT_DATATYPES = {
    "Stem length": Real(bounds=(25.45, 140.0)),
    "Stem angle": Real(bounds=(-30.0, 32.55)),
    "Crank length": Real(bounds=(142.5, 185.0)),
    "Headset spacers": Real(bounds=(0.0, 50.0)),
    "Handlebar style": Choice(options=(1, 2))
}

FRAMED_MATERIAL_COLUMNS = ['Material=Steel', 'Material=Aluminum', 'Material=Titanium']


def unscaled_framed_dataset():
    x, y, x_scaler, y_scaler = load_augmented_framed_dataset()
    x_columns = x.columns
    x_index = x.index
    return pd.DataFrame(x_scaler.inverse_transform(x), columns=x_columns, index=x_index)


UNSCALED_FRAMED = unscaled_framed_dataset()


def _map_framed_column(result, column):
    bounds = (UNSCALED_FRAMED[column].quantile(0.01), UNSCALED_FRAMED[column].quantile(0.99))
    print(f"Mapped {column} to Real with bounds {bounds}")
    result.append(Real(bounds=bounds))


def is_categorical(column_name: str):
    if column_name in ['Material', 'SSB_Include', 'CSB_Include']:
        return True
    if column_name in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        return True
    return False


def map_combined_datatypes(dataframe: pd.DataFrame) -> List[Variable]:
    result = []
    for column in dataframe.columns:
        if is_categorical(column):
            options = tuple(CATEGORY_MAPS[column].keys())
            print(f"Mapping {column} to choice with options {options}")
            result.append(Choice(options=options))
        elif column in FRAMED_COLUMNS:
            _map_framed_column(result, column)
        elif column in CLIPS_COLUMNS:
            mapped_datatype = map_column(dataframe[column])
            result.append(mapped_datatype)
        else:
            result.append(BIKE_FIT_DATATYPES[column])
            print(f"Mapped bike fit column {column}")
    return result


def _get_or_default(row_value: str, category_map: dict):
    try:
        return category_map[row_value]
    except KeyError:
        print(f"Failed to get value from {category_map} by {row_value}. Defaulting to random.")
        return random.choice(list(category_map.values()))


class CombinedDataset:
    """The hierarchy here is FRAMED -> CLIPS -> BIKE_FIT"""

    def __init__(self, data: pd.DataFrame):
        self._data = data

    @classmethod
    def build_from_three(cls, framed_style: pd.DataFrame,
                         clips_style: pd.DataFrame,
                         bike_fit_style: pd.DataFrame):
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
        cls._reverse_framed_one_hot_encoding(result)
        cls._reverse_clips_one_hot_encoding(result)
        cls._append_bike_fit(result, bike_fit_style)
        cls._category_to_numeric(result)
        result = result.astype('float64')
        return CombinedDataset(result)

    def get_combined(self):
        return self._data.copy(deep=True)

    def get_as_framed(self) -> pd.DataFrame:
        result = self._data.copy(deep=True)
        self._numeric_to_category(result)
        self._framed_one_hot_encode(result)
        return pd.DataFrame(result, columns=FRAMED_COLUMNS)

    def get_as_bike_fit(self) -> pd.DataFrame:
        result = self._get_bike_fit_from_clips()
        self._append_unique_bike_fit_columns(result)
        self._append_bike_fit_defaults(result)
        return pd.DataFrame(result, columns=BIKE_FIT_COLUMNS)

    def _get_bike_fit_from_clips(self):
        return (self.get_as_clips().copy(deep=True).rename(
            columns=self._reverse_map(FRAMED_TO_CLIPS_IDENTICAL)).rename(
            columns=self._reverse_map(FRAMED_TO_CLIPS_UNITS)))

    def get_as_clips(self) -> pd.DataFrame:
        data = self._data.copy(deep=True)
        self._numeric_to_category(data)
        self._framed_one_hot_encode(data)
        self._to_clips_material(data)
        self._append_clips_one_hot_encoded(data)
        self._to_millimeter_columns(data)
        self._to_clips_identical(data)
        return pd.DataFrame(data, columns=CLIPS_COLUMNS)

    def _to_clips_material(self, data: pd.DataFrame):
        for material in CLIPS_IGNORED_MATERIAL:
            data[material] = 0
        framed_material_columns = self._get_framed_material_columns()
        print(f"Found framed material columns {framed_material_columns}")
        for material_column in framed_material_columns:
            clips_column = self._to_clips_material_column(material_column)
            data[clips_column] = data.get(material_column, 0)
        data.drop(columns=framed_material_columns, inplace=True, errors='ignore')
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
        return FRAMED_CLIPS_INTERSECTION_COLUMNS

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

    @classmethod
    def _reverse_map(cls, one_to_one_map: dict):
        return {
            v: k for k, v in one_to_one_map.items()
        }

    @staticmethod
    def _append_bike_fit_defaults(data):
        CombinedDataset._add_if_missing(data, 'Stem length', 100)
        CombinedDataset._add_if_missing(data, 'Stem angle', -5)
        CombinedDataset._add_if_missing(data, 'Handlebar style', 2)
        CombinedDataset._add_if_missing(data, 'Crank length', 170)
        CombinedDataset._add_if_missing(data, 'Headset spacers', 10)

    @staticmethod
    def _add_if_missing(data: pd.DataFrame, column_name: str, default_value: float):
        if column_name not in list(data.columns):
            print(f"Setting column {column_name} to default value {default_value}")
            noise = np.random.rand(len(data)) * 0.1 * default_value
            print(f"Adding noise to column {column_name} with average {np.average(noise)}")
            data[column_name] = (default_value + noise)

    def _append_unique_bike_fit_columns(self, result: pd.DataFrame):
        self._append_bike_fit(result, self._data)

    @classmethod
    def _append_bike_fit(cls, result: pd.DataFrame, bike_fit_style: pd.DataFrame):
        for column in UNIQUE_BIKE_FIT_COLUMNS:
            if column in bike_fit_style.columns:
                result[column] = bike_fit_style[column]

    @classmethod
    def _map_framed_material(cls, result: pd.DataFrame):

        def extract_material(row: pd.Series):
            for c in FRAMED_MATERIAL_COLUMNS:
                if int(row.loc[c]) == 1:
                    return c.split('=')[1]

        result['Material'] = result.apply(func=extract_material, axis=1)

    def _get_framed_material_columns(self):
        return FRAMED_MATERIAL_COLUMNS

    @classmethod
    def _reverse_framed_one_hot_encoding(cls, result: pd.DataFrame):
        cls._reverse_one_hot_encoding(
            result,
            lambda column_name: 'Material' in column_name,
            separator='='
        )

    @classmethod
    def _reverse_clips_one_hot_encoding(cls, result: pd.DataFrame):
        cls._reverse_one_hot_encoding(
            result,
            column_finder=lambda column_name: 'OHCLASS' in column_name.upper(),
            separator=CLIPS_ONE_HOT_ENCODING_SEP
        )

    def _framed_one_hot_encode(self, result: pd.DataFrame) -> None:
        self._append_one_hot_encoded(
            data=result,
            columns=['Material'],
            prefix_sep='='
        )

    def _append_clips_one_hot_encoded(self, data: pd.DataFrame):
        self._append_one_hot_encoded(
            data=data,
            columns=[c for c in ONE_HOT_ENCODED_CLIPS_COLUMNS if 'material' != c.lower()],
            prefix_sep=CLIPS_ONE_HOT_ENCODING_SEP
        )

    def _append_one_hot_encoded(self, data: pd.DataFrame, columns: List[str], prefix_sep: str):
        for column in columns:
            encoded_dataframe = get_encoded_columns(data, column_name=column, prefix_sep=prefix_sep)
            for encoded_column in encoded_dataframe.columns:
                data[encoded_column] = encoded_dataframe[encoded_column].values
        data.drop(columns=columns, inplace=True)

    @classmethod
    def _reverse_one_hot_encoding(cls,
                                  result: pd.DataFrame,
                                  column_finder: Callable[[str], bool],
                                  separator: str,
                                  ):
        columns_to_drop = [c for c in result.columns if column_finder(c)]
        reversed_encoding = reverse_one_hot_encoding(result,
                                                     column_finder=column_finder,
                                                     separator=separator)
        for c in reversed_encoding.columns:
            result[c] = reversed_encoding[c].values
        result.drop(columns=columns_to_drop, inplace=True)

    @classmethod
    def _category_to_numeric(cls, result: pd.DataFrame):
        for column in result.columns:
            if column in CATEGORY_MAPS.keys():
                category_map = cls._reverse_map(CATEGORY_MAPS[column])
                result[column] = result[column].apply(lambda row_value: category_map[row_value])

    def _numeric_to_category(self, result):
        for column in result.columns:
            if column in CATEGORY_MAPS.keys():
                to_category_map = (CATEGORY_MAPS[column])
                result[column] = result[column].apply(lambda row_value: _get_or_default(row_value, to_category_map))


class OriginalCombinedDataset:
    def __init__(self):
        x_framed = self._get_framed_unscaled()
        clips = self._get_clips_corrected_index()
        bike_fit = self._get_bike_fit_corrected_index()
        intersection = self._index_intersection(clips, x_framed, bike_fit)
        self._combined = CombinedDataset.build_from_three(framed_style=x_framed.loc[intersection],
                                                          clips_style=clips.loc[intersection],
                                                          bike_fit_style=bike_fit.loc[intersection])

    def _index_intersection(self, clips: pd.DataFrame, x_framed: pd.DataFrame, bike_fit: pd.DataFrame):
        idx_intersection = set(x_framed.index).intersection(set(clips.index))
        return list(idx_intersection.intersection(set(bike_fit.index)))

    def _get_clips_corrected_index(self):
        clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
        clips.index = [str(idx) for idx in clips.index]
        return clips

    def _get_framed_unscaled(self):
        x_framed, y_framed, x_scaler, y_scaler = load_augmented_framed_dataset()
        return pd.DataFrame(x_scaler.inverse_transform(x_framed), columns=x_framed.columns, index=x_framed.index)

    def get_combined_dataset(self) -> CombinedDataset:
        return self._combined

    def _get_bike_fit_corrected_index(self):
        bike_fit = (pd.read_csv(resource_path('bike_vector_df_with_id.csv'),
                                index_col='Bike ID')
                    .drop(columns=['Unnamed: 0']))
        bike_fit.index = [str(idx) for idx in bike_fit.index]
        return bike_fit
