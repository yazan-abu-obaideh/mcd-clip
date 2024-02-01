import numpy as np


def deconvert(df, dataset=""):
    if dataset == "":
        if "RDERD" in df.columns:
            df["ERD rear"] = df["Wheel diameter rear"] - df["RDERD"]
            df.drop(["RDERD"], axis=1, inplace=True)
        if "FDERD" in df.columns:
            df["ERD front"] = df["Wheel diameter front"] - df["FDERD"]
            df.drop(["FDERD"], axis=1, inplace=True)
        if "RDBSD" in df.columns:
            df["BSD rear"] = df["Wheel diameter rear"] - df["RDBSD"]
            df.drop(["RDBSD"], axis=1, inplace=True)
        if "FDBSD" in df.columns:
            df["BSD front"] = df["Wheel diameter front"] - df["FDBSD"]
            df.drop(["FDBSD"], axis=1, inplace=True)
        df["nCHAINSTAYOFFSET"] = df["CHAINSTAYOFFSET"]
        df["nCHAINSTAYAUXrearDIAMETER"] = df["CHAINSTAYAUXrearDIAMETER"]
        df["nChain stay horizontal diameter"] = df["Chain stay horizontal diameter"]
        df["nChain stay position on BB"] = df["Chain stay position on BB"]
        df["nChain stay taper"] = df["Chain stay taper"]
        df["nChain stay back diameter"] = df["Chain stay back diameter"]
        df["nChain stay vertical diameter"] = df["Chain stay vertical diameter"]
        df["nSeat stay junction0"] = df["Seat stay junction0"]
        df["nSeat stay bottom diameter"] = df["Seat stay bottom diameter"]
        df["nSEATSTAY_HF"] = df["SEATSTAY_HF"]
        df["nSSTopZOFFSET"] = df["SSTopZOFFSET"]
        df["nSEATSTAY_HR"] = df["SEATSTAY_HR"]
        df["nSEATSTAYTAPERLENGTH"] = df["SEATSTAYTAPERLENGTH"]

    if dataset in ["micro", "clip_s"]:
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

    if dataset in ["mini"]:
        pass
    if dataset in ["clip", "clip_s"]:
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
