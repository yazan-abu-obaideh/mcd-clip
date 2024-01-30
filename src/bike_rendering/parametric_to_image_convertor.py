import pandas as pd

from bike_rendering.bike_xml_handler import BikeXmlHandler
from resource_utils import resource_path


class ParametricToImageConvertor:
    def to_image(self, bike: pd.Series):
        pass


if __name__ == "__main__":
    handler = BikeXmlHandler()
    with open(resource_path("PlainRoadBikeStandardized.txt")) as file:
        handler.set_xml(file.read())

    data = pd.read_csv(resource_path("clip_sBIKED_processed.csv"), index_col=0)
    keys_in_file = set(handler.get_all_keys())
    keys_in_dataset = set(data.columns)
    not_in_file = list(keys_in_dataset.difference(keys_in_file))
    is_oh_class = ["OHCLASS" in k for k in not_in_file]
    for i in range(len(is_oh_class)):
        if "OHCLASS" not in not_in_file[i]:
            print(not_in_file[i])
