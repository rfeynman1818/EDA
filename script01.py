import os
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point, Polygon

NS = {'ns': 'urn:SICD:1.3.0'}

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def extract_polygon_from_sicd(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        corners = []
        for icp in root.findall(".//ns:ImageCorners/ns:ICP", namespaces=NS):
            lat = float(icp.find("ns:Lat", NS).text)
            lon = float(icp.find("ns:Lon", NS).text)
            corners.append((lon, lat))  # shapely expects (lon, lat)
        if len(corners) != 4:
            return None
        return Polygon(corners)
    except Exception as e:
        logger.warning(f"Failed to parse {xml_path}: {e}")
        return None


def file_matches_location(xml_path, lat, lon):
    poly = extract_polygon_from_sicd(xml_path)
    if poly is None:
        return False
    return poly.contains(Point(lon, lat))


def scan_file_for_location(file_path, location):
    try:
        if file_path.suffix.lower() == '.xml' and 'sicd' in file_path.name.lower():
            if file_matches_location(file_path, location['latitude_deg'], location['longitude_deg']):
                return file_path
    except Exception as e:
        logger.warning(f"Error processing {file_path}: {e}")
    return None


def find_matching_files(directory, locations, output_path="sicd_matches.txt"):
    directory = Path(directory)
    all_files = list(directory.rglob("*.xml"))

    with open(output_path, "w") as f_out:
        for location in locations:
            logger.info(f"Checking Location: {location['name']}")
            matches = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(scan_file_for_location, f, location) for f in all_files]
                for future in futures:
                    result = future.result()
                    if result:
                        matches.append(result)

            if matches:
                f_out.write(f"Location: {location['name']}\n")
                for match in matches:
                    f_out.write(str(match) + "\n")
                f_out.write("\n")
            logger.info(f"Found {len(matches)} matches for {location['name']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Directory to scan for SICD XML files")
    args = parser.parse_args()

    locations = [
        {"name": "Kant", "latitude_deg": 42.9, "longitude_deg": 74.85},
        {"name": "Nalchik", "latitude_deg": 43.51278, "longitude_deg": 43.63611}
    ]

    find_matching_files(args.directory, locations)
