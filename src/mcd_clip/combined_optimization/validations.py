import numpy as np
import pandas as pd


def validate_combined_seat_height(bike_df: pd.DataFrame) -> pd.DataFrame:
    return _validate_seat_height(bike_df, bike_df["ST Length"] * 1000)


def validate_clips_seat_height(bike_df: pd.DataFrame) -> pd.DataFrame:
    return _validate_seat_height(bike_df, bike_df["Seat tube length"])


def _validate_seat_height(bike_df: pd.DataFrame, seat_tube_column):
    # TODO: flipped because of how MCD handles constraints (this is not intuitive and should be fixed)
    difference = bike_df["Saddle height"] - seat_tube_column
    within_range = np.logical_and(difference < 125, difference > -125)
    handles_seat_post = np.logical_and(
        within_range,
        bike_df['Seatpost LENGTH'] - seat_tube_column < 85
    )
    return 1 - pd.DataFrame(handles_seat_post).astype('int32')
