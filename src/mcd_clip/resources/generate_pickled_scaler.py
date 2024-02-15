import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcd_clip.datasets.columns_constants import CLIPS_COLUMNS
import dill

ALL_PARAMETRIC_CSV_PATH = ""  # insert here!

parameters = pd.read_csv(ALL_PARAMETRIC_CSV_PATH, index_col=0)

print(set(parameters.columns) == set(CLIPS_COLUMNS))
print(len(parameters.columns))
scaler = StandardScaler()
scaler.fit_transform(parameters)
print(f"{scaler.mean_=}")
print(f"{type(scaler.mean_)=}")
print(f"{scaler.with_std=}")
with open('scaler.pk', 'wb') as file:
    dill.dump(scaler, file)
