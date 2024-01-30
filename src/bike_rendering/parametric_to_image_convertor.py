import pandas as pd

from bike_rendering.bike_xml_handler import BikeXmlHandler
from resource_utils import resource_path

ONE_HOT_ENCODED_VALUES = ['MATERIAL', 'Dropout spacing style',
                          'Head tube type', 'BELTorCHAIN',
                          'bottle SEATTUBE0 show', 'RIM_STYLE front',
                          'RIM_STYLE rear', 'Handlebar style',
                          'bottle DOWNTUBE0 show', 'Stem kind',
                          'Fork type', 'Top tube type']


class ParametricToImageConvertor:
    def to_image(self, bike: pd.Series):
        pass


def to_sRgb():
    pass


if __name__ == "__main__":
    handler = BikeXmlHandler()
    with open(resource_path("PlainRoadBikeStandardized.txt")) as file:
        handler.set_xml(file.read())
    entries_dict = handler.get_entries_dict()
    first_bike = pd.read_csv(resource_path("clip_sBIKED_processed.csv"), index_col=0).iloc[0]
    first_bike_dict = first_bike.to_dict()
    og_entries = entries_dict.copy()

    one_hot_encoded = []

    for k, v in first_bike_dict.items():
        if k in entries_dict:
            entries_dict[k] = str(v)
        if "OH" in k:
            print(f"OH in {k}")
            one_hot_encoded.append(k.split("OHCLASS")[0].strip())
    print(list(set(one_hot_encoded)))
