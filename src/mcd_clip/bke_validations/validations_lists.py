from typing import Callable

import numpy as np
import pandas as pd

__CLIPS_VALIDATIONS_RAW = [
    lambda df: df["Saddle height"] < df["Seat tube length"] + 40,
    lambda df: df["Saddle height"] > (df["Seat tube length"] + df["Seatpost LENGTH"] + 30),
    lambda df: df["BSD rear"] < df["ERD rear"],
    lambda df: df["BSD front"] < df["ERD front"],
    lambda df: df["Head tube lower extension2"] >= df["Head tube length textfield"],
    lambda df: ((df["Head tube upper extension2"] + df["Head tube lower extension2"]) >= df[
        'Head tube length textfield']),
    lambda df: df["CS textfield"] <= 0,
]

__MULTIPLIER = 1000
__COMBINED_VALIDATIONS_RAW = [
    lambda df: df["Saddle height"] < (df["ST Length"] * __MULTIPLIER) + 40,
    lambda df: df["Saddle height"] > ((df["ST Length"] * __MULTIPLIER) + df["Seatpost LENGTH"] + 30),
    lambda df: df["BSD rear"] < df["ERD rear"],
    lambda df: df["BSD front"] < df["ERD front"],
    lambda df: df["HT LX"] >= df["HT Length"],
    lambda df: ((df["HT UX"] + df["HT LX"]) >= df['HT Length']),
    lambda df: df["CS textfield"] <= 0,
]


def _wrap_function(validation_function: Callable):
    def wrapped_function(designs: pd.DataFrame):
        try:
            validation_result = validation_function(designs).astype("int32")
            print(f"Validation successful percent invalid [{np.sum(validation_result) / len(designs)}%]")
            return validation_result
        except KeyError as e:
            print(f"Validation function failed {e}...")
            return pd.DataFrame(np.zeros(shape=(len(designs), 1)))

    return wrapped_function


def build_clips_validations():
    return [_wrap_function(validation_function=validation_function)
            for validation_function in __CLIPS_VALIDATIONS_RAW]


def build_combined_validations():
    return [_wrap_function(validation_function=validation_function)
            for validation_function in __COMBINED_VALIDATIONS_RAW]


CLIPS_VALIDATION_FUNCTIONS = build_clips_validations()
COMBINED_VALIDATION_FUNCTIONS = build_combined_validations()
