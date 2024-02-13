import numpy as np
import pandas as pd

from mcd_clip.bke_validations.df_check import DATAFRAME_VALIDATION_FUNCTIONS
from mcd_clip.resource_utils import resource_path

clips = pd.read_csv(resource_path('clip_sBIKED_processed.csv'), index_col=0)
result = pd.DataFrame()
for fun in DATAFRAME_VALIDATION_FUNCTIONS:
    result = pd.concat([result, fun(clips)], axis=1)
print(result)
print(np.sum(result))
