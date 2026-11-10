from dataclasses import dataclass


@dataclass(frozen=True)
class Satellite:
    key: str
    label: str
    instrument: str
    pixel: str


SATELLITES: dict[str, Satellite] = {
    "terra": Satellite("terra", "Terra", "MODIS", "1 km"),
    "aqua": Satellite("aqua", "Aqua", "MODIS", "1 km"),
    "snpp": Satellite("snpp", "Suomi-NPP", "VIIRS", "375 m"),
    "noaa20": Satellite("noaa20", "NOAA-20", "VIIRS", "375 m"),
    "noaa21": Satellite("noaa21", "NOAA-21", "VIIRS", "375 m"),
    "landsat": Satellite("landsat", "Landsat 8/9", "OLI", "30 m"),
    "other": Satellite("other", "Other source", "Unknown", "unknown"),
}
ORDER = list(SATELLITES)


def satellite_of(source_id: str, platform: str | None = None) -> str:
    """Map a FIREBRIDGE source id (e.g. 'modis-aqua-c61-standard') to one satellite, merging collections."""
    text = f"{source_id} {platform or ''}".lower()
    for needle, key in (
        ("terra", "terra"),
        ("aqua", "aqua"),
        ("snpp", "snpp"),
        ("noaa20", "noaa20"),
        ("noaa21", "noaa21"),
        ("n20", "noaa20"),
        ("n21", "noaa21"),
        ("landsat", "landsat"),
    ):
        if needle in text:
            return key
    return "other"


def label(key: str) -> str:
    return SATELLITES.get(key, SATELLITES["other"]).label
