import os.path
import uuid

import pandas as pd

from mcd_clip.bike_rendering.bikeCad_renderer import RenderingService
from mcd_clip.bike_rendering.bike_xml_handler import BikeXmlHandler
from mcd_clip.resource_utils import resource_path

RENDERING_SERVICE = RenderingService(1)

STANDARD_BIKE_RESOURCE = "PlainRoadBikeStandardized.txt"

ONE_HOT_ENCODED_VALUES = ['MATERIAL', 'Dropout spacing style',
                          'Head tube type', 'BELTorCHAIN',
                          'bottle SEATTUBE0 show', 'RIM_STYLE front',
                          'RIM_STYLE rear', 'Handlebar style',
                          'bottle DOWNTUBE0 show', 'Stem kind',
                          'Fork type', 'Top tube type']


def _run_result_path(result_file):
    return os.path.join(os.path.dirname(__file__), "..", "run-results", result_file)


class ParametricToImageConvertor:
    def to_image(self, bike: pd.Series):
        handler = BikeXmlHandler()

        self._read_standard_bike_xml(handler)
        decoded_values = one_hot_decode(bike)
        bike_dict = bike.to_dict()
        all_bike_keys = handler.get_all_keys()
        bike_dict.update(decoded_values)
        self._update_non_encoded_values(all_bike_keys, bike_dict, handler)

        bike_uuid = uuid.uuid4()

        with open(_run_result_path(f"bike_{bike_uuid}.txt"), "w") as file:
            file.write(handler.get_content_string())
        xml = handler.get_content_string()
        with open(_run_result_path(f"bike_{bike_uuid}.png"), "wb") as image_file:
            image_file.write(RENDERING_SERVICE.render(xml))

    def _read_standard_bike_xml(self, handler):
        with open(resource_path(STANDARD_BIKE_RESOURCE)) as file:
            handler.set_xml(file.read())

    def _update_non_encoded_values(self, all_bike_keys, bike_dict, handler):
        for k, v in bike_dict.items():
            if k in all_bike_keys:
                print(f"Updating {k}")
                handler.add_or_update(k, str(v))


def to_sRgb():
    pass


def one_hot_decode(bike: pd.Series) -> dict:
    result = {}
    for encoded_value in ONE_HOT_ENCODED_VALUES:
        for column in bike.index:
            if encoded_value in column and bike[column] == 1:
                result[encoded_value] = column.split('OHCLASS:')[1].strip()
    return result


if __name__ == "__main__":
    convertor = ParametricToImageConvertor()
    convertor.to_image(pd.read_csv(resource_path("clip_sBIKED_processed.csv"), index_col=0).iloc[5])
