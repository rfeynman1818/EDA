# LLH-to-Pixel Projection Investigation Cookbook

## Objective
Investigate performance issues in the current ATR (Automatic Target Recognition) model by checking for errors in the LLH (Latitude, Longitude, Height) to Slant pixel coordinate projection on Spotlight images from the Unsupervised dataset.

## Prerequisites
- Access to the ATR model codebase
- Unsupervised dataset with Spotlight SAR images
- SARPy library for robust SAR processing
- Additional geospatial libraries for validation

## Step 1: Environment Setup

### 1.1 Required Dependencies
```bash
# Core Python dependencies
pip install numpy matplotlib scipy pandas

# SAR Processing - Primary library for this investigation
pip install sarpy

# Additional geospatial libraries for validation and comparison
pip install pyproj gdal rasterio

# Optional: OpenCV for additional image processing (if needed)
pip install opencv-python
```

### 1.2 Import Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

# SARPy imports - Primary SAR processing capabilities
from sarpy.io.complex.utils import get_reader
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.complex.base import SICDTypeReader
from sarpy.geometry import geocoords
from sarpy.geometry.point_projection import image_to_ground, ground_to_image

# Additional geospatial libraries for validation
import pyproj
from pyproj import Transformer, CRS
try:
    import gdal
    from osgeo import osr
except ImportError:
    print("GDAL not available - using pyproj only")

# Optional OpenCV import
try:
    import cv2
except ImportError:
    print("OpenCV not available - skipping")
```

### 1.3 Configure Logging
```python
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

## Step 2: Data Preparation

### 2.1 Load SAR Images with SARPy
```python
def load_sar_images_with_sarpy(dataset_path):
    """
    Load SAR images using SARPy reader
    Supports SICD, SIDD, and other SAR formats
    """
    sar_files = []
    
    # Common SAR file extensions
    extensions = ['*.ntf', '*.nitf', '*.sicd', '*.sidd', '*.tiff', '*.tif']
    
    for ext in extensions:
        sar_files.extend(Path(dataset_path).glob(f"**/{ext}"))
    
    valid_readers = []
    for file_path in sar_files:
        try:
            reader = get_reader(str(file_path))
            if reader is not None:
                valid_readers.append((file_path, reader))
                logger.info(f"Successfully loaded SAR file: {file_path.name}")
        except Exception as e:
            logger.warning(f"Could not load {file_path.name}: {e}")
    
    logger.info(f"Found {len(valid_readers)} valid SAR images in dataset")
    return valid_readers

dataset_path = "/path/to/unsupervised/dataset"
sar_images = load_sar_images_with_sarpy(dataset_path)
```

### 2.2 Extract SAR Metadata with SARPy
```python
def extract_sar_metadata(reader):
    """
    Extract comprehensive SAR metadata using SARPy
    """
    try:
        if hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None:
            sicd = reader.sicd_meta
            
            metadata = {
                'sicd_meta': sicd,
                'image_formation': sicd.ImageFormation,
                'grid_type': sicd.Grid,
                'geo_data': sicd.GeoData,
                'position': sicd.Position,
                'radar_collection': sicd.RadarCollection,
                'image_size': (sicd.ImageData.NumRows, sicd.ImageData.NumCols),
                'pixel_spacing': (sicd.Grid.Row.SS, sicd.Grid.Col.SS),
                'center_frequency': sicd.RadarCollection.TxFrequency.Min if sicd.RadarCollection else None,
                'collect_start': sicd.Timeline.CollectStart if sicd.Timeline else None,
                'scp_llh': sicd.GeoData.SCP.LLH if sicd.GeoData and sicd.GeoData.SCP else None
            }
            
            return metadata
        else:
            logger.warning("No SICD metadata available")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting SAR metadata: {e}")
        return None
```

## Step 3: LLH-to-Pixel Projection with SARPy

### 3.1 Current ATR Model Projection Function
```python
def current_llh_to_pixel_projection(lat, lon, height, reader):
    """
    Current implementation from ATR model
    TODO: Replace with actual ATR model projection code
    
    This is a placeholder - replace with your ATR model's projection method
    """
    try:
        # Placeholder implementation using basic SARPy
        if hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None:
            sicd = reader.sicd_meta
            
            # Convert to ECEF first, then to image coordinates
            llh = np.array([lat, lon, height])
            ecef = geocoords.geodetic_to_ecef([llh])[0]
            
            # Use SARPy's ground_to_image function
            image_coords = ground_to_image(ecef, sicd, block_size=1)
            
            if image_coords is not None and len(image_coords) > 0:
                row, col = image_coords[0]
                return int(col), int(row)  # Return as (x, y) pixel coordinates
        
        return None, None
    
    except Exception as e:
        logger.error(f"ATR projection error: {e}")
        return None, None
```

### 3.2 SARPy Reference Projection Function
```python
def sarpy_llh_to_pixel_projection(lat, lon, height, reader):
    """
    Reference implementation using SARPy's robust projection methods
    """
    try:
        if hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None:
            sicd = reader.sicd_meta
            
            # Method 1: Direct ground-to-image projection
            llh_point = np.array([lat, lon, height])
            
            # Convert LLH to ECEF coordinates
            ecef_coords = geocoords.geodetic_to_ecef([llh_point])
            
            # Use SARPy's ground_to_image projection
            image_coords = ground_to_image(ecef_coords, sicd, 
                                         use_structure_coa=True, 
                                         block_size=1)
            
            if image_coords is not None and len(image_coords) > 0:
                row, col = image_coords[0]
                
                # Ensure coordinates are within image bounds
                num_rows, num_cols = sicd.ImageData.NumRows, sicd.ImageData.NumCols
                
                if 0 <= row < num_rows and 0 <= col < num_cols:
                    return int(col), int(row)  # Return as (x, y)
                else:
                    logger.debug(f"Projected coordinates outside image bounds: ({col}, {row})")
                    return int(col), int(row)  # Still return for error analysis
            
        return None, None
        
    except Exception as e:
        logger.error(f"SARPy projection error: {e}")
        return None, None
```

### 3.3 Validation with Round-trip Projection
```python
def validate_projection_accuracy(lat, lon, height, reader):
    """
    Validate projection accuracy using round-trip projection
    LLH -> Pixel -> LLH to check consistency
    """
    try:
        if hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None:
            sicd = reader.sicd_meta
            
            # Forward projection: LLH -> Pixel
            pixel_x, pixel_y = sarpy_llh_to_pixel_projection(lat, lon, height, reader)
            
            if pixel_x is not None and pixel_y is not None:
                # Reverse projection: Pixel -> LLH
                image_coords = np.array([[pixel_y, pixel_x]])  # SARPy uses (row, col)
                
                # Use SARPy's image_to_ground function
                ecef_coords = image_to_ground(image_coords, sicd, 
                                            projection_type='HAE', 
                                            heights=height)
                
                if ecef_coords is not None and len(ecef_coords) > 0:
                    # Convert ECEF back to LLH
                    llh_result = geocoords.ecef_to_geodetic(ecef_coords)[0]
                    
                    # Calculate round-trip errors
                    lat_error = abs(lat - llh_result[0])
                    lon_error = abs(lon - llh_result[1])
                    height_error = abs(height - llh_result[2])
                    
                    return {
                        'pixel_coords': (pixel_x, pixel_y),
                        'recovered_llh': llh_result,
                        'lat_error': lat_error,
                        'lon_error': lon_error,
                        'height_error': height_error,
                        'total_error': np.sqrt(lat_error**2 + lon_error**2)
                    }
            
        return None
        
    except Exception as e:
        logger.error(f"Round-trip validation error: {e}")
        return None
```

## Step 4: Error Detection and Analysis

### 4.1 Generate Test Points from SAR Scene
```python
def generate_test_llh_points_from_sar(reader, num_points=100):
    """
    Generate test LLH points intelligently based on SAR scene coverage
    """
    try:
        if hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None:
            sicd = reader.sicd_meta
            
            # Get scene corner coordinates from SICD metadata
            if sicd.GeoData and sicd.GeoData.ImageCorners:
                corners = sicd.GeoData.ImageCorners
                corner_coords = []
                
                for corner in [corners.ICP1, corners.ICP2, corners.ICP3, corners.ICP4]:
                    if corner and corner.LLH:
                        corner_coords.append([corner.LLH.Lat, corner.LLH.Lon, corner.LLH.HAE or 0])
                
                if len(corner_coords) == 4:
                    # Calculate bounding box
                    lats = [coord[0] for coord in corner_coords]
                    lons = [coord[1] for coord in corner_coords]
                    heights = [coord[2] for coord in corner_coords]
                    
                    min_lat, max_lat = min(lats), max(lats)
                    min_lon, max_lon = min(lons), max(lons)
                    min_height, max_height = min(heights), max(heights)
                    
                    # Generate random points within scene bounds
                    test_lats = np.random.uniform(min_lat, max_lat, num_points)
                    test_lons = np.random.uniform(min_lon, max_lon, num_points)
                    test_heights = np.random.uniform(max(0, min_height), max_height + 1000, num_points)
                    
                    return list(zip(test_lats, test_lons, test_heights))
            
            # Fallback: use scene center point if corners not available
            if sicd.GeoData and sicd.GeoData.SCP and sicd.GeoData.SCP.LLH:
                scp = sicd.GeoData.SCP.LLH
                center_lat, center_lon, center_height = scp.Lat, scp.Lon, scp.HAE or 0
                
                # Generate points around scene center
                lat_range = 0.01  # ~1km at equator
                lon_range = 0.01
                height_range = 500
                
                test_lats = np.random.uniform(center_lat - lat_range, center_lat + lat_range, num_points)
                test_lons = np.random.uniform(center_lon - lon_range, center_lon + lon_range, num_points)
                test_heights = np.random.uniform(center_height, center_height + height_range, num_points)
                
                return list(zip(test_lats, test_lons, test_heights))
                
    except Exception as e:
        logger.error(f"Error generating test points: {e}")
    
    # Ultimate fallback: generate global test points
    logger.warning("Using fallback global test points")
    lats = np.random.uniform(-85, 85, num_points)
    lons = np.random.uniform(-180, 180, num_points)
    heights = np.random.uniform(0, 1000, num_points)
    
    return list(zip(lats, lons, heights))
```

### 4.2 SAR-Specific Projection Error Analysis
```python
def analyze_sar_projection_errors(file_path, reader, test_points):
    """
    Comprehensive projection error analysis for SAR data
    """
    metadata = extract_sar_metadata(reader)
    if not metadata:
        logger.warning(f"No metadata available for {file_path.name}")
        return None
    
    results = []
    
    for i, (lat, lon, height) in enumerate(test_points):
        try:
            # Current ATR projection
            current_x, current_y = current_llh_to_pixel_projection(lat, lon, height, reader)
            
            # SARPy reference projection
            ref_x, ref_y = sarpy_llh_to_pixel_projection(lat, lon, height, reader)
            
            # Round-trip validation
            roundtrip_result = validate_projection_accuracy(lat, lon, height, reader)
            
            # Calculate errors if both projections succeeded
            if current_x is not None and ref_x is not None:
                error_x = abs(current_x - ref_x)
                error_y = abs(current_y - ref_y)
                euclidean_error = np.sqrt(error_x**2 + error_y**2)
                
                # Check image bounds
                num_rows, num_cols = metadata['image_size']
                in_bounds_current = (0 <= current_x < num_cols and 0 <= current_y < num_rows)
                in_bounds_ref = (0 <= ref_x < num_cols and 0 <= ref_y < num_rows)
                
                result = {
                    'point_id': i,
                    'lat': lat, 'lon': lon, 'height': height,
                    'current_x': current_x, 'current_y': current_y,
                    'ref_x': ref_x, 'ref_y': ref_y,
                    'error_x': error_x, 'error_y': error_y,
                    'euclidean_error': euclidean_error,
                    'in_bounds_current': in_bounds_current,
                    'in_bounds_ref': in_bounds_ref,
                    'image_size': f"{num_rows}x{num_cols}",
                    'pixel_spacing': metadata['pixel_spacing']
                }
                
                # Add round-trip validation results
                if roundtrip_result:
                    result.update({
                        'roundtrip_lat_error': roundtrip_result['lat_error'],
                        'roundtrip_lon_error': roundtrip_result['lon_error'], 
                        'roundtrip_height_error': roundtrip_result['height_error'],
                        'roundtrip_total_error': roundtrip_result['total_error']
                    })
                
                results.append(result)
                
        except Exception as e:
            logger.warning(f"Error processing point {i}: {e}")
            continue
    
    if results:
        return pd.DataFrame(results)
    return None
```

## Step 5: Comprehensive Testing

### 5.1 Run Analysis on SAR Image Collection
```python
def run_comprehensive_sar_analysis(sar_images, num_test_points=50):
    """
    Run projection error analysis on multiple SAR images using SARPy
    """
    all_results = []
    
    for i, (file_path, reader) in enumerate(sar_images[:10]):  # Analyze first 10 images
        logger.info(f"Analyzing SAR image {i+1}/{min(10, len(sar_images))}: {file_path.name}")
        
        try:
            # Extract metadata first to validate the image
            metadata = extract_sar_metadata(reader)
            if not metadata:
                logger.warning(f"Skipping {file_path.name} - no valid metadata")
                continue
            
            # Generate test points specific to this SAR scene
            test_points = generate_test_llh_points_from_sar(reader, num_test_points)
            
            # Run projection error analysis
            results_df = analyze_sar_projection_errors(file_path, reader, test_points)
            
            if results_df is not None and not results_df.empty:
                results
