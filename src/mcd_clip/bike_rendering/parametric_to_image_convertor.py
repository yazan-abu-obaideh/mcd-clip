import uuid

import pandas as pd

from mcd_clip.bike_rendering.bikeCad_renderer import RenderingService
from mcd_clip.bike_rendering.bike_xml_handler import BikeXmlHandler
from mcd_clip.bike_rendering.clips_to_bcad import deconvert
from mcd_clip.resource_utils import resource_path, run_result_path

RENDERING_SERVICE = RenderingService(1)

STANDARD_BIKE_RESOURCE = "PlainRoadBikeStandardized.txt"

ONE_HOT_ENCODED_VALUES = ['MATERIAL', 'Dropout spacing style',
                          'Head tube type', 'BELTorCHAIN',
                          'bottle SEATTUBE0 show', 'RIM_STYLE front',
                          'RIM_STYLE rear', 'Handlebar style',
                          'bottle DOWNTUBE0 show', 'Stem kind',
                          'Fork type', 'Top tube type']


class ParametricToImageConvertor:
    def to_image(self, bike: pd.Series, path_prefix="design"):
        handler = BikeXmlHandler()

        bike_complete = deconvert(pd.DataFrame(data=[bike])).iloc[0]

        self._read_standard_bike_xml(handler)
        decoded_values = one_hot_decode(bike_complete)
        bike_dict = bike_complete.to_dict()
        all_bike_keys = handler.get_all_keys()
        bike_dict.update(decoded_values)
        self._update_non_encoded_values(all_bike_keys, bike_dict, handler)

        bike_uuid = uuid.uuid4()

        with open(run_result_path(f"{path_prefix}_bike_{bike_uuid}.txt"), "w") as file:
            file.write(handler.get_content_string())
        xml = handler.get_content_string()
        with open(run_result_path(f"{path_prefix}_bike_{bike_uuid}.png"), "wb") as image_file:
            image_file.write(RENDERING_SERVICE.render(xml))

    def _read_standard_bike_xml(self, handler):
        with open(resource_path(STANDARD_BIKE_RESOURCE)) as file:
            handler.set_xml(file.read())

    def _update_non_encoded_values(self, all_bike_keys, bike_dict, handler):
        for k, v in bike_dict.items():
            if k in all_bike_keys:
                print(f"Updating {k} with value {v}")
                handler.add_or_update(k, str(v))


def one_hot_decode(bike: pd.Series) -> dict:
    result = {}
    for encoded_value in ONE_HOT_ENCODED_VALUES:
        for column in bike.index:
            if encoded_value in column and bike[column] == 1:
                result[encoded_value] = column.split('OHCLASS:')[1].strip()
    return result
