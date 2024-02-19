import attrs
import pandas as pd

from mcd_clip.bike_rendering.bikeCad_renderer import RenderingService
from mcd_clip.bike_rendering.bike_xml_handler import BikeXmlHandler
from mcd_clip.bike_rendering.clips_to_bcad import clips_to_cad
from mcd_clip.datasets.columns_constants import ONE_HOT_ENCODED_CLIPS_COLUMNS
from mcd_clip.resource_utils import resource_path

RENDERING_SERVICE = RenderingService(1)

STANDARD_BIKE_RESOURCE = "PlainRoadBikeStandardized.txt"


@attrs.define
class RenderingResult:
    image: bytes
    bike_xml: str


class ParametricToImageConvertor:
    def to_image(self, target_bike: pd.Series) -> RenderingResult:
        xml_handler = self._build_xml_handler()
        target_dict = self._to_cad_dict(target_bike)
        self._update_values(xml_handler, target_dict)
        updated_xml = xml_handler.get_content_string()
        return RenderingResult(image=(RENDERING_SERVICE.render(updated_xml)), bike_xml=updated_xml)

    def _build_xml_handler(self):
        xml_handler = BikeXmlHandler()
        self._read_standard_bike_xml(xml_handler)
        return xml_handler

    def _to_cad_dict(self, bike):
        bike_complete = clips_to_cad(pd.DataFrame(data=[bike])).iloc[0]
        decoded_values = one_hot_decode(bike_complete)
        bike_dict = bike_complete.to_dict()
        bike_dict.update(decoded_values)
        return self._remove_encoded_values(bike_dict)

    def _read_standard_bike_xml(self, handler):
        with open(resource_path(STANDARD_BIKE_RESOURCE)) as file:
            handler.set_xml(file.read())

    def _update_values(self, handler, bike_dict):
        num_updated = 0
        for k, v in bike_dict.items():
            parsed = self._parse(v)
            if parsed is not None:
                num_updated += 1
                self._update_value(parsed, handler, k)
        print(f"{num_updated=}")

    def _parse(self, v):
        handled = self._handle_numeric(v)
        handled = self._handle_bool(str(handled))
        return handled

    def _update_value(self, handled, xml_handler, k):
        print(f"Updating {k} with value {handled}")
        xml_handler.add_or_update(k, handled)

    def _handle_numeric(self, v):
        if str(v).lower() == 'nan':
            return None
        if type(v) in [int, float]:
            v = int(v)
        return v

    def _handle_bool(self, param):
        if param.lower().title() in ['True', 'False']:
            return param.lower()
        return param

    def _remove_encoded_values(self, bike_dict: dict) -> dict:
        to_delete = []
        for k, _ in bike_dict.items():
            for encoded_key in ONE_HOT_ENCODED_CLIPS_COLUMNS:
                if "OHCLASS" in k and encoded_key in k:
                    print(f"Deleting key {k}")
                    to_delete.append(k)
        return {
            k: v for k, v in bike_dict.items() if k not in to_delete
        }


def one_hot_decode(bike: pd.Series) -> dict:
    result = {}
    for encoded_value in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        for column in bike.index:
            if encoded_value in column and bike[column] == 1:
                result[encoded_value] = column.split('OHCLASS:')[1].strip()
    return result
