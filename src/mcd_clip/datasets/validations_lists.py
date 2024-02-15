from typing import Callable, List

import numpy as np
import pandas as pd

__MULTIPLIER = 1000
__COMBINED_VALIDATIONS_RAW = [
    lambda df: df["Saddle height"] < (df["ST Length"] * __MULTIPLIER) + 40,
    lambda df: df["Saddle height"] > ((df["ST Length"] * __MULTIPLIER) + df["Seatpost LENGTH"] + 30),
    lambda df: df["BSD rear"] < df["ERD rear"],
    lambda df: df["BSD front"] < df["ERD front"],
    lambda df: df["HT LX"] >= df["HT Length"],
    lambda df: ((df["HT UX"] + df["HT LX"]) >= df['HT Length']),
]

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


def _wrap_function(validation_function: Callable):
    def wrapped_function(designs: pd.DataFrame):
        try:
            validation_result = validation_function(designs).astype("int32")
            print(f"Validation result: fraction invalid [{np.sum(validation_result) / len(designs)}]")
            return validation_result
        except KeyError as e:
            print(f"Validation function failed {e}...")
            return pd.DataFrame(np.zeros(shape=(len(designs), 1)))

    return wrapped_function


def _build_validations(raw_validations_list: List[Callable]):
    return [_wrap_function(validation_function=validation_function)
            for validation_function in raw_validations_list]


CLIPS_VALIDATION_FUNCTIONS = _build_validations(__CLIPS_VALIDATIONS_RAW)
COMBINED_VALIDATION_FUNCTIONS = _build_validations(__COMBINED_VALIDATIONS_RAW)
