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
        num_updated = self._update_non_encoded_values(all_bike_keys, bike_dict, handler)

        print(f"{num_updated=}")

        bike_uuid = uuid.uuid4()

        updated_xml = handler.get_content_string()
        with open(run_result_path(f"{path_prefix}_bike_{bike_uuid}.txt"), "w") as file:
            file.write(updated_xml)
        with open(run_result_path(f"{path_prefix}_bike_{bike_uuid}.png"), "wb") as image_file:
            image_file.write(RENDERING_SERVICE.render(updated_xml))

    def _read_standard_bike_xml(self, handler):
        with open(resource_path(STANDARD_BIKE_RESOURCE)) as file:
            handler.set_xml(file.read())

    def _update_non_encoded_values(self, all_bike_keys, bike_dict, handler):
        num_updated = 0
        for k, v in bike_dict.items():
            if k in all_bike_keys:
                num_updated += 1
                if str(v).lower() == 'nan':
                    continue
                if type(v) in [int, float]:
                    print(v)
                    v = int(v)
                handled = self._handle_bool(str(v))
                print(f"Updating {k} with value {handled}")
                handler.add_or_update(k, handled)
        return num_updated

    def _handle_bool(self, param):
        if param.lower().title() in ['True', 'False']:
            return param.lower()
        return param


def one_hot_decode(bike: pd.Series) -> dict:
    result = {}
    for encoded_value in ONE_HOT_ENCODED_VALUES:
        for column in bike.index:
            if encoded_value in column and bike[column] == 1:
                result[encoded_value] = column.split('OHCLASS:')[1].strip()
    return result
