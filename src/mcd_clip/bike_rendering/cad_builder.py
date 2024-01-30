from mcd_clip.bike_rendering.bike_xml_handler import BikeXmlHandler
from mcd_clip.resource_utils import resource_path

OPTIMIZED_TO_CAD = {
    "ST Angle": "Seat angle",
    "HT Length": "Head tube length textfield",
    "HT Angle": "Head angle",
    "HT LX": "Head tube lower extension2",
    'Stack': 'Stack',
    "ST Length": "Seat tube length",
    "Seatpost LENGTH": "Seatpost LENGTH",
    "Saddle height": "Saddle height",
    "Stem length": "Stem length",
    "Crank length": "Crank length",
    "Headset spacers": "Headset spacers",
    "Stem angle": "Stem angle",
    "Handlebar style": "Handlebar style",
}


def _get_valid_seed_bike(seed_image_id):
    if str(seed_image_id) not in [str(_) for _ in range(1, 14)]:
        raise ValueError(f"Invalid seed bike ID [{seed_image_id}]")
    return f"bike{seed_image_id}.bcad"


class BikeCadFileBuilder:
    def build_cad_from_object(self, bike_object, seed_bike_id) -> str:
        xml_handler = BikeXmlHandler()
        self._load_seed_xml(xml_handler, seed_bike_id)
        for response_key, cad_key in OPTIMIZED_TO_CAD.items():
            self._update_xml(xml_handler, cad_key, bike_object[response_key])
        # self._update_xml(xml_handler, "Display RIDER", "true")
        return xml_handler.get_content_string()

    def _load_seed_xml(self, xml_handler, seed_image_id):
        with open(resource_path("PlainRoadBikeStandardized.txt"), "r") as file:
            xml_handler.set_xml(file.read())

    def _update_xml(self, xml_handler, cad_key, desired_value):
        entry = xml_handler.find_entry_by_key(cad_key)
        if entry:
            xml_handler.update_entry_value(entry, str(desired_value))
        else:
            xml_handler.add_new_entry(cad_key, str(desired_value))
