"""
Enhanced functions for extraction of SAR metadata from SICD XML files
"""

# Standard Library Imports
import os
import logging
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime

# External Imports
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from sarpy.io.complex.sicd import SICDType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

class EnhancedSARMetadataExtractor:
    """
    Enhanced extractor for SAR metadata from SICD XML files with comprehensive data extraction.
    """

    def __init__(self, directory_path: Path = None):
        """
        Initialize the Enhanced SAR Metadata Extractor.

        Args:
            directory_path (Path): Path to the directory containing SICD XML files.
        """
        self.directory_path = Path(directory_path) if directory_path else None
        self.metadata = {}
        self.metadata_array = None
        self.column_names = []

    def parse_sicd_metadata(self) -> dict:
        """
        Parses all SICD XML metadata files within the specified directory.

        Returns:
            dict: Dictionary containing metadata for each file.
        """
        if not self.directory_path or not self.directory_path.exists():
            raise ValueError("Valid directory path required")
            
        metadata = {}
        xml_files = list(self.directory_path.rglob("*.xml"))
        
        _logger.info(f"Found {len(xml_files)} XML files")
        
        for file_path in tqdm(xml_files, desc="Processing XML files"):
            if 'SICD' in file_path.name or self._is_sicd_file(file_path):
                file_metadata = self.extract_metadata(file_path)
                if file_metadata is not None:
                    metadata[file_path.name] = file_metadata

        self.metadata = metadata
        self._create_numpy_arrays()
        return metadata

    def extract_metadata(self, file_path: Path) -> dict:
        """
        Extracts comprehensive metadata fields using sarpy and direct XML parsing.

        Args:
            file_path (Path): Path to the SICD XML file.

        Returns:
            dict: Dictionary containing metadata for the file.
        """
        _logger.info(f"Extracting metadata from file: {file_path}")
        
        try:
            # First try with sarpy
            metadata = self._extract_with_sarpy(file_path)
            
            # Supplement with direct XML parsing for additional fields
            xml_metadata = self._parse_xml_directly(file_path)
            metadata.update(xml_metadata)
            
            # Add computed fields
            self._add_computed_fields(metadata)
            
            _logger.info(f"Finished extracting metadata from file: {file_path}")
            return metadata
            
        except Exception as e:
            _logger.error(f"Error parsing file {file_path}: {e}")
            # Try fallback to direct XML parsing
            try:
                metadata = self._parse_xml_directly(file_path)
                self._add_computed_fields(metadata)
                return metadata
            except:
                return None

    def _extract_with_sarpy(self, file_path: Path) -> dict:
        """Extract metadata using sarpy library."""
        metadata = {}
        
        try:
            sicd_type = SICDType.from_xml_file(file_path)
            
            # Extract Collection Info
            collection_info = getattr(sicd_type, "CollectionInfo", None)
            if collection_info:
                metadata['CollectorName'] = getattr(collection_info, 'CollectorName', 'Unknown')
                metadata['CoreName'] = getattr(collection_info, 'CoreName', 'Unknown')
                metadata['CollectType'] = getattr(collection_info, 'CollectType', 'Unknown')
                metadata['ModeID'] = getattr(collection_info, 'ModeID', 'Unknown')

                radar_mode = getattr(collection_info, "RadarMode", None)
                if radar_mode and hasattr(radar_mode, "ModeType"):
                    metadata["ModeType"] = radar_mode.ModeType
                else:
                    metadata["ModeType"] = "Unknown"

            # Extract Image Data
            image_data = getattr(sicd_type, "ImageData", None)
            if image_data:
                metadata['PixelType'] = getattr(image_data, 'PixelType', 'Unknown')
                metadata['NumRows'] = getattr(image_data, 'NumRows', 0)
                metadata['NumCols'] = getattr(image_data, 'NumCols', 0)

            # Extract Incidence Angle
            if hasattr(sicd_type, 'SCPCOA') and hasattr(sicd_type.SCPCOA, 'IncidenceAng'):
                metadata['IncidenceAng'] = sicd_type.SCPCOA.IncidenceAng

            # Extract Geospatial Information
            try:
                geo_data = sicd_type.GeoData
                scp = geo_data.SCP
                llh = scp.LLH
                metadata['Lat'] = llh.Lat
                metadata['Lon'] = llh.Lon
                metadata['HAE'] = llh.HAE
                metadata['SCP_Lat'] = llh.Lat
                metadata['SCP_Lon'] = llh.Lon
                metadata['SCP_HAE'] = llh.HAE
            except AttributeError as e:
                _logger.warning(f"Missing geospatial data in {file_path}: {e}")

        except Exception as e:
            _logger.warning(f"Sarpy extraction failed for {file_path}: {e}")
        
        return metadata

    def _parse_xml_directly(self, xml_file_path: Path) -> dict:
        """
        Parse XML file directly for additional fields not covered by sarpy.
        """
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # Define namespace - handle both with and without namespace
            ns = {'sicd': 'urn:SICD:1.3.0'} if root.tag.startswith('{') else {}
            
            metadata = {}
            
            # Image Creation
            image_creation = self._find_element(root, 'ImageCreation', ns)
            if image_creation is not None:
                metadata['Application'] = self._get_text(image_creation, 'Application', ns)
                datetime_str = self._get_text(image_creation, 'DateTime', ns)
                if datetime_str:
                    metadata['DateTime'] = datetime_str
                    metadata['Date'] = datetime_str.split('T')[0]
                    metadata['Time'] = datetime_str.split('T')[1].replace('Z', '') if 'T' in datetime_str else datetime_str
                metadata['Site'] = self._get_text(image_creation, 'Site', ns)
            
            # SCPCOA additional fields
            scpcoa = self._find_element(root, 'SCPCOA', ns)
            if scpcoa is not None:
                metadata['TwistAng'] = self._get_float(scpcoa, 'TwistAng', ns)
                metadata['SlopeAng'] = self._get_float(scpcoa, 'SlopeAng', ns)
                metadata['AzimAng'] = self._get_float(scpcoa, 'AzimAng', ns)
            
            # Image corners
            geo_data = self._find_element(root, 'GeoData', ns)
            if geo_data is not None:
                corners = self._find_element(geo_data, 'ImageCorners', ns)
                if corners is not None:
                    corner_coords = []
                    corner_data = {}
                    for icp in corners.findall(self._tag_with_ns('ICP', ns)):
                        index = icp.get('index', '')
                        lat = self._get_float(icp, 'Lat', ns)
                        lon = self._get_float(icp, 'Lon', ns)
                        if lat is not None and lon is not None:
                            corner_coords.append((lat, lon))
                            corner_data[f'Corner_{index}_Lat'] = lat
                            corner_data[f'Corner_{index}_Lon'] = lon
                    
                    metadata.update(corner_data)
                    if len(corner_coords) >= 3:
                        metadata['Corner_Coords'] = corner_coords
            
            # Extract filename info
            filename = xml_file_path.name
            metadata['Filename'] = filename
            metadata['FilePath'] = str(xml_file_path)
            
            # Parse filename for additional info
            self._parse_filename_info(metadata, filename)
            
            return metadata
            
        except Exception as e:
            _logger.error(f"Direct XML parsing failed for {xml_file_path}: {e}")
            return {}

    def _find_element(self, parent, tag, ns):
        """Find element with or without namespace."""
        if ns and 'sicd' in ns:
            return parent.find(f"sicd:{tag}", ns)
        else:
            return parent.find(tag)

    def _tag_with_ns(self, tag, ns):
        """Get tag with namespace if available."""
        if ns and 'sicd' in ns:
            return f"sicd:{tag}"
        return tag

    def _get_text(self, element, tag, ns):
        """Get text from XML element."""
        elem = self._find_element(element, tag, ns)
        return elem.text if elem is not None and elem.text else None

    def _get_int(self, element, tag, ns):
        """Get integer from XML element."""
        text = self._get_text(element, tag, ns)
        try:
            return int(text) if text is not None else None
        except (ValueError, TypeError):
            return None

    def _get_float(self, element, tag, ns):
        """Get float from XML element."""
        text = self._get_text(element, tag, ns)
        try:
            return float(text) if text is not None else None
        except (ValueError, TypeError):
            return None

    def _parse_filename_info(self, metadata, filename):
        """Parse filename for additional metadata."""
        if 'CAPELLA' in filename.upper():
            parts = filename.split('_')
            if len(parts) >= 3:
                metadata['Mission'] = parts[0]
                metadata['Processing_Level'] = parts[-2] if len(parts) > 2 else 'Unknown'
        elif 'ICEYE' in filename.upper():
            parts = filename.split('_')
            if len(parts) >= 3:
                metadata['Mission'] = 'ICEYE'
                metadata['Satellite'] = parts[0] if parts[0].startswith('ICEYE') else 'Unknown'

    def _add_computed_fields(self, metadata):
        """Add computed fields to metadata."""
        # Image size in megapixels
        if metadata.get('NumRows') and metadata.get('NumCols'):
            metadata['ImageSize_MP'] = (metadata['NumRows'] * metadata['NumCols']) / 1e6
            metadata['AspectRatio'] = metadata['NumCols'] / metadata['NumRows']
        
        # Approximate area from corner coordinates
        if 'Corner_Coords' in metadata:
            metadata['Approx_Area_km2'] = self._calculate_polygon_area(metadata['Corner_Coords'])

    def _calculate_polygon_area(self, coordinates):
        """Calculate approximate area of polygon in km²."""
        try:
            if len(coordinates) >= 3:
                poly = Polygon([(lon, lat) for lat, lon in coordinates])
                area_deg_sq = poly.area
                area_km_sq = area_deg_sq * (111 ** 2)
                return area_km_sq
        except Exception as e:
            _logger.warning(f"Could not calculate area: {e}")
            return None

    def _is_sicd_file(self, file_path: Path) -> bool:
        """Check if file is a SICD XML file by examining content."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return 'SICD' in root.tag or any('SICD' in elem.tag for elem in root.iter())
        except:
            return False

    def _create_numpy_arrays(self):
        """Create numpy arrays from metadata dictionary for easier analysis."""
        if not self.metadata:
            return
        
        # Collect all unique keys
        all_keys = set()
        for metadata in self.metadata.values():
            all_keys.update(metadata.keys())
        
        # Filter out complex objects and keep only simple data types
        simple_keys = []
        for key in all_keys:
            sample_value = None
            for metadata in self.metadata.values():
                if key in metadata:
                    sample_value = metadata[key]
                    break
            
            if sample_value is not None and isinstance(sample_value, (str, int, float, bool)):
                simple_keys.append(key)
        
        self.column_names = sorted(simple_keys)
        
        # Create data matrix
        data_matrix = []
        filenames = []
        
        for filename, metadata in self.metadata.items():
            row = []
            for key in self.column_names:
                value = metadata.get(key)
                if isinstance(value, str):
                    # Keep strings as is for now
                    row.append(value)
                elif isinstance(value, (int, float)):
                    row.append(float(value))
                elif isinstance(value, bool):
                    row.append(float(value))
                else:
                    row.append(None)
            data_matrix.append(row)
            filenames.append(filename)
        
        self.metadata_array = np.array(data_matrix, dtype=object)
        self.filenames = np.array(filenames)

    def get_numeric_data(self):
        """Get numeric data as numpy arrays."""
        if self.metadata_array is None:
            return None, None
        
        # Find numeric columns
        numeric_cols = []
        numeric_indices = []
        
        for i, col_name in enumerate(self.column_names):
            # Check if column contains numeric data
            col_data = self.metadata_array[:, i]
            is_numeric = True
            for val in col_data:
                if val is not None:
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        is_numeric = False
                        break
            
            if is_numeric:
                numeric_cols.append(col_name)
                numeric_indices.append(i)
        
        if not numeric_indices:
            return None, None
        
        # Extract numeric data
        numeric_data = self.metadata_array[:, numeric_indices].astype(float)
        # Handle None values
        numeric_data = np.where(numeric_data == None, np.nan, numeric_data)
        
        return numeric_data, numeric_cols

    def get_summary_statistics(self) -> dict:
        """Get summary statistics of the metadata."""
        if not self.metadata:
            return {}
        
        # Count collectors
        collectors = {}
        radar_modes = {}
        dates = []
        
        for metadata in self.metadata.values():
            collector = metadata.get('CollectorName', 'Unknown')
            collectors[collector] = collectors.get(collector, 0) + 1
            
            mode = metadata.get('ModeType', 'Unknown')
            radar_modes[mode] = radar_modes.get(mode, 0) + 1
            
            if 'DateTime' in metadata:
                dates.append(metadata['DateTime'])
        
        # Geographic extent
        lats = [m.get('SCP_Lat') for m in self.metadata.values() if m.get('SCP_Lat') is not None]
        lons = [m.get('SCP_Lon') for m in self.metadata.values() if m.get('SCP_Lon') is not None]
        
        geographic_extent = {}
        if lats and lons:
            geographic_extent = {
                'lat_min': min(lats),
                'lat_max': max(lats),
                'lon_min': min(lons),
                'lon_max': max(lons),
                'center_lat': np.mean(lats),
                'center_lon': np.mean(lons)
            }
        
        # Image statistics
        numeric_data, numeric_cols = self.get_numeric_data()
        image_stats = {}
        if numeric_data is not None:
            for i, col in enumerate(numeric_cols):
                col_data = numeric_data[:, i]
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    image_stats[col] = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'count': len(valid_data)
                    }
        
        stats = {
            'total_files': len(self.metadata),
            'collectors': collectors,
            'radar_modes': radar_modes,
            'date_range': {
                'start': min(dates) if dates else None,
                'end': max(dates) if dates else None
            },
            'geographic_extent': geographic_extent,
            'image_statistics': image_stats
        }
        
        return stats

    def save_metadata_csv(self, output_path: str = "sicd_metadata.csv"):
        """Save metadata to CSV file using basic file operations."""
        if not self.metadata:
            _logger.warning("No metadata available to save")
            return
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                # Write header
                if self.column_names:
                    f.write('Filename,' + ','.join(self.column_names) + '\n')
                    
                    # Write data
                    for i, filename in enumerate(self.filenames):
                        row_data = [str(filename)]
                        for j in range(len(self.column_names)):
                            value = self.metadata_array[i, j] if self.metadata_array is not None else ''
                            row_data.append(str(value) if value is not None else '')
                        f.write(','.join(row_data) + '\n')
                else:
                    # Fallback: write metadata dictionary directly
                    all_keys = set()
                    for metadata in self.metadata.values():
                        all_keys.update(metadata.keys())
                    
                    header = ['Filename'] + sorted(all_keys)
                    f.write(','.join(header) + '\n')
                    
                    for filename, metadata in self.metadata.items():
                        row = [filename]
                        for key in sorted(all_keys):
                            value = metadata.get(key, '')
                            if isinstance(value, (list, dict)):
                                value = str(value)
                            row.append(str(value))
                        f.write(','.join(row) + '\n')
            
            _logger.info(f"Metadata saved to {output_path}")
        except Exception as e:
            _logger.error(f"Error saving CSV: {e}")