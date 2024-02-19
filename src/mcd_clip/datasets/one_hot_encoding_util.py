from typing import Callable

import pandas as pd


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
    data = data.copy(deep=True)
    columns = [str(column) for column in data.columns if column_finder(str(column))]
    return pd.from_dummies(data[columns].astype('int32'), sep=separator)
