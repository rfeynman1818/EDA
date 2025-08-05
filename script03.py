import os
import numpy as np
from sarpy.io.complex.sicd import open as open_sicd
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

input_dir = '/path/to/nitf_files'
output_dir = '/path/to/output_tifs'
os.makedirs(output_dir, exist_ok=True)

def sicd_to_grd_tif(nitf_path, output_path):
    reader = open_sicd(nitf_path)
    sicd = reader.get_sicds_as_tuple()[0]
    data = np.abs(reader[0])

    if sicd.GeoData is None or sicd.GeoData.SCP is None:
        raise ValueError("GeoData missing, cannot georeference.")

    scp = sicd.GeoData.SCP
    pixel_spacing = abs(sicd.Grid.Row.SS), abs(sicd.Grid.Col.SS)
    scp_lat, scp_lon = scp.Lat, scp.Lon
    height, width = data.shape
    transform = from_origin(scp_lon - width * pixel_spacing[1] / 2,
                            scp_lat + height * pixel_spacing[0] / 2,
                            pixel_spacing[1], pixel_spacing[0])

    with rasterio.open(output_path, 'w', driver='GTiff',
                       height=height, width=width,
                       count=1, dtype=data.dtype,
                       crs='EPSG:4326',
                       transform=transform) as dst:
        dst.write(data, 1)

converted_files = []
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.ntf', '.nitf')):
        nitf_path = os.path.join(input_dir, filename)
        out_name = os.path.splitext(filename)[0] + '_GRD.tif'
        output_path = os.path.join(output_dir, out_name)
        try:
            sicd_to_grd_tif(nitf_path, output_path)
            print(f'Converted: {filename}')
            converted_files.append(output_path)
        except Exception as e:
            print(f'Failed: {filename} | {e}')

def visualize_geotiff(path):
    with rasterio.open(path) as src:
        img = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    plt.figure(figsize=(8, 6))
    plt.imshow(10 * np.log10(img + 1e-3), extent=extent, cmap='gray', origin='upper')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(os.path.basename(path))
    plt.colorbar(label='Backscatter (dB)')
    plt.show()

# Visualize all converted GeoTIFFs
for tif_path in converted_files:
    visualize_geotiff(tif_path)
