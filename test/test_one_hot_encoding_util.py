import unittest
from typing import Callable

import pandas as pd

from mcd_clip.resource_utils import resource_path


def get_encoded_columns(data: pd.DataFrame,
                        column_name: str,
                        prefix_sep: str) -> pd.DataFrame:
    data = data.copy(deep=True)
    data[column_name] = pd.Categorical(data[column_name], categories=list(data[column_name].unique()))
    return pd.get_dummies(data[column_name], prefix=column_name, prefix_sep=prefix_sep)


def reverse_one_hot_encoding(data: pd.DataFrame,
                             column_finder: Callable[[str], bool],
                             separator: str
                             ) -> pd.DataFrame:
    columns = [str(column) for column in data.columns if column_finder(str(column))]
    return pd.from_dummies(data[columns], sep=separator)


class OneHotEncodingUtilTest(unittest.TestCase):
    def setUp(self):
        self.framed_original = pd.read_csv(resource_path('all_structural_data_aug.csv'), index_col=0)

    def test_framed_encode(self):
        print(self.framed_original['Material'])
        encoded_materials = get_encoded_columns(self.framed_original,
                                                column_name='Material',
                                                prefix_sep='='
                                                )
        print(encoded_materials)
        revered_encoding = reverse_one_hot_encoding(encoded_materials,
                                                    column_finder=lambda column_name: 'material' in column_name.lower(),
                                                    separator='=')
        print(revered_encoding)
