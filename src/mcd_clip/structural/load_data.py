# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:05:43 2022

@author: Lyle
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcd_clip.resource_utils import resource_path

ALL_STRUCTURAL_DATASET = resource_path('all_structural_data_aug.csv')


def one_hot_encode_material(data: pd.DataFrame):
    data = data.copy()
    # One-hot encode the materials
    data["Material"] = pd.Categorical(data["Material"], categories=["Steel", "Aluminum", "Titanium"])
    mats_oh = pd.get_dummies(data["Material"], prefix="Material=", prefix_sep="")
    data.drop(["Material"], axis=1, inplace=True)
    data = pd.concat([mats_oh, data], axis=1)
    return data


def load_augmented_framed_dataset(to_magnitudes=False) -> (
        tuple)[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    reg_data = pd.read_csv(ALL_STRUCTURAL_DATASET, index_col=0)

    x = reg_data.iloc[:, :-11]

    x = one_hot_encode_material(x)

    x, x_scaler = scale(x)
    y = reg_data.iloc[:, -11:-1]

    for col in ['Sim 1 Safety Factor', 'Sim 3 Safety Factor']:
        y[col] = 1 / y[col]
        y.rename(columns={col: col + " (Inverted)"}, inplace=True)

    convert_to_magnitudes(y, to_magnitudes)

    y, y_scaler = scale(y)

    return x, y, x_scaler, y_scaler


def convert_to_magnitudes(y, to_magnitudes):
    if to_magnitudes:
        for col in ['Sim 1 Dropout X Disp.', 'Sim 1 Dropout Y Disp.', 'Sim 1 Bottom Bracket X Disp.',
                    'Sim 1 Bottom Bracket Y Disp.', 'Sim 2 Bottom Bracket Z Disp.', 'Sim 3 Bottom Bracket Y Disp.',
                    'Sim 3 Bottom Bracket X Rot.', 'Model Mass']:
            y[col] = [np.abs(val) for val in y[col].values]
            y.rename(columns={col: col + " Magnitude"}, inplace=True)


def scale(v):
    v_scaler = StandardScaler()
    v_scaler.fit(v)
    v_scaled_values = v_scaler.transform(v)
    new_v = pd.DataFrame(v_scaled_values, columns=v.columns, index=v.index)
    return new_v, v_scaler
