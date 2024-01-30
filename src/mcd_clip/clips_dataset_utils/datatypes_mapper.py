import pandas as pd
from pymoo.core.variable import Real, Integer, Choice

from mcd_clip.resource_utils import resource_path

_MAPPINGS = {
    "float64": lambda lower_bound, upper_bound: Real(bounds=(lower_bound, upper_bound)),
    "int64": lambda lower_bound, upper_bound: Integer(bounds=(lower_bound, upper_bound)),
    "bool": lambda x1, x2: Choice(options=(0, 1))
}

_NAME_TO_TYPE = pd.read_csv(resource_path("clip_sBIKED_processed_datatypes.csv"),
                            index_col=0)


def map_column(column: pd.Series):
    column_datatype = _NAME_TO_TYPE.loc[column.name].values[0]
    return _MAPPINGS[column_datatype](column.min(), column.max())
