import pandas as pd


def validate_combined_seat_height(bike_df: pd.DataFrame) -> pd.DataFrame:
    return _validate_seat_height(bike_df, "ST Length")


def validate_clips_seat_height(bike_df: pd.DataFrame) -> pd.DataFrame:
    return _validate_seat_height(bike_df, "Seat tube length")


def _validate_seat_height(bike_df: pd.DataFrame, seat_tube_column: str):
    # TODO: flipped because of how MCD handles constraints (this is not intuitive and should be fixed)
    return pd.DataFrame((bike_df["Saddle height"] - bike_df[seat_tube_column]) < 65).astype('int32')
