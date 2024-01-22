import os.path

import pandas as pd
from pymoo.core.variable import Real, Integer, Binary

MAPPINGS = {
    "float64": lambda lower_bound, upper_bound: Real(bounds=(lower_bound, upper_bound)),
    "int64": lambda lower_bound, upper_bound: Integer(bounds=(lower_bound, upper_bound)),
    "bool": lambda x1, x2: Binary()
}

path = os.path.join(os.path.dirname(__file__), "resources", "clip_sBIKED_processed_datatypes.csv")
name_to_type = pd.read_csv(path, index_col=0)


def map_column(column: pd.Series):
    column_datatype = name_to_type.loc[column.name].values[0]
    return MAPPINGS[column_datatype](column.min(), column.max())
