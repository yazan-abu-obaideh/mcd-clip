import numpy as np
import pandas as pd


def clips_to_cad(df: pd.DataFrame):
    if "csd" in df.columns:
        df["Chain stay back diameter"] = df["csd"]
        df["Chain stay vertical diameter"] = df["csd"]
    if "ssd" in df.columns:
        df["SEATSTAY_HR"] = df["ssd"]
        df["Seat stay bottom diameter"] = df["ssd"]
    if "ttd" in df.columns:
        df["Top tube rear diameter"] = df["ttd"]
        df["Top tube rear dia2"] = df["ttd"]
        df["Top tube front diameter"] = df["ttd"]
        df["Top tube front dia2"] = df["ttd"]
    if "dtd" in df.columns:
        df["Down tube rear diameter"] = df["dtd"]
        df["Down tube rear dia2"] = df["dtd"]
        df["Down tube front diameter"] = df["dtd"]
        df["Down tube front dia2"] = df["dtd"]
    for idx in df.index:
        Stack = df.at[idx, "Stack"]
        HTL = df.at[idx, "Head tube length textfield"]
        HTLX = df.at[idx, "Head tube lower extension2"]
        HTA = df.at[idx, "Head angle"] * np.pi / 180
        BBD = df.at[idx, "BB textfield"]
        DTL = df.at[idx, "DT Length"]
        DTJY = Stack - (HTL - HTLX) * np.sin(HTA)
        DTJX = np.sqrt(DTL ** 2 - DTJY ** 2)
        FWX = DTJX + (DTJY - BBD) / np.tan(HTA)
        FCD = np.sqrt(FWX ** 2 + BBD ** 2)
        df.at[idx, "FCD textfield"] = FCD
    df.drop(["DT Length"], axis=1, inplace=True)
    for column in list(df.columns):
        if column.endswith("R_RGB"):
            r = df[column].values
            g = df[column.replace("R_RGB", "G_RGB")].values
            b = df[column.replace("R_RGB", "B_RGB")].values
            df.drop(column, axis=1, inplace=True)
            df.drop(column.replace("R_RGB", "G_RGB"), axis=1, inplace=True)
            df.drop(column.replace("R_RGB", "B_RGB"), axis=1, inplace=True)
            val = r * (2 ** 16) + g * (2 ** 8) + b - (2 ** 24)
            df[column.replace("R_RGB", "sRGB")] = val
    return df.copy()
