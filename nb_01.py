import os, json, yaml, csv
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import shape
from sklearn.manifold import TSNE
from skimage import measure
from sarpy.io.complex import open_complex


# ------------------------------------------------------
# 1. Load and Parse SICD Metadata
# ------------------------------------------------------
directory_path = input("Enter path to directory with SICD XML files: ")
metadata_extractor = SARMetadataExtractor(directory_path)
metadata = metadata_extractor.parse_sicd_metadata()
print(f"Parsed {len(metadata)} metadata entries from SICD XML files")

# ------------------------------------------------------
# 2. Export Metadata to CSV
# ------------------------------------------------------
metadata_extractor.save_metadata_csv("exported_metadata.csv")
print("Metadata exported to 'exported_metadata.csv'")

# ------------------------------------------------------
# 3. Preview Sample Metadata Dictionary
# ------------------------------------------------------
first_key = list(metadata.keys())[0]
metadata[first_key]

# ------------------------------------------------------
# 4. Summary Statistics
# ------------------------------------------------------
summary = metadata_extractor.get_summary_statistics()
print(yaml.dump(summary, default_flow_style=False))

# ------------------------------------------------------
# 5. Convert Metadata to Numpy Arrays
# ------------------------------------------------------
metadata_array, column_names = metadata_extractor.get_numeric_data()
print("Shape:", metadata_array.shape)
print("Columns:", column_names)

# ------------------------------------------------------
# 6. Initialize SARMetadataAnalyzer
# ------------------------------------------------------
sar_metadata_analyzer = SARMetadataAnalyzer(directory_path, metadata_extractor)

# ------------------------------------------------------
# 7. Visualize Image Type Counts
# ------------------------------------------------------
image_types = sar_metadata_analyzer.count_image_types()
sar_metadata_analyzer.plot_image_type_counts(image_types)

# ------------------------------------------------------
# 8. Image Count by Radar Mode
# ------------------------------------------------------
sar_metadata_analyzer.plot_image_count_by_mode(metadata)

# ------------------------------------------------------
# 9. Incidence Angle Distribution by Radar Mode
# ------------------------------------------------------
modes = list(set(m.get("ModeType", "Unknown") for m in metadata.values()))
for mode in modes:
    sar_metadata_analyzer.plot_incidence_angle_distributions(metadata, mode)

# ------------------------------------------------------
# 10. Plot Arbitrary Numeric Metadata Attribute
# ------------------------------------------------------
sar_metadata_analyzer.plot_metadata_distribution(metadata, "AspectRatio", "Aspect Ratio Distribution")
sar_metadata_analyzer.plot_metadata_distribution(metadata, "ImageSize_MPx", "Image Size in Megapixels")

# ------------------------------------------------------
# 11. Geospatial AOI Visualization
# ------------------------------------------------------
lat_min = float(input("Enter minimum latitude: "))
lat_max = float(input("Enter maximum latitude: "))
lon_min = float(input("Enter minimum longitude: "))
lon_max = float(input("Enter maximum longitude: "))
sar_metadata_analyzer.plot_geospatial_data(metadata, lat_min, lat_max, lon_min, lon_max)

# ------------------------------------------------------
# 12. Vehicle EDA (Object-Level)
# ------------------------------------------------------
print("\nRunning Vehicle-Level EDA...")
data_dir = directory_path
label_dir = input("Enter path to vehicle label GeoJSON files: ")

records = []
for img_name in tqdm(os.listdir(data_dir)):
    if not (img_name.endswith(".sicd") or img_name.endswith(".nitf")):
        continue
    scene_id = os.path.splitext(img_name)[0]
    label_path = os.path.join(label_dir, scene_id + ".geojson")
    if not os.path.exists(label_path):
        continue

    img, meta = open_complex(os.path.join(data_dir, img_name)).read_chip(), None
    img = np.abs(img)

    with open(label_path) as f:
        gj = json.load(f)

    for ftr in gj["features"]:
        geom = shape(ftr["geometry"])
        mask = measure.grid_points_in_poly(img.shape, np.array(geom.exterior.coords))
        vals = img[mask]
        if vals.size == 0:
            continue
        mean_i = np.mean(vals)
        area = np.sum(mask)
        records.append((scene_id, mean_i, area))

records = np.array(records, dtype=object)
means = records[:, 1].astype(float)
areas = records[:, 2].astype(float)

plt.hist(areas, bins=50)
plt.title("Vehicle Bounding Box Area Distribution")
plt.xlabel("Pixels²"); plt.ylabel("Count")
plt.show()

plt.hist(10 * np.log10(means + 1e-6), bins=50)
plt.title("Vehicle Mean Backscatter (dB)")
plt.xlabel("dB"); plt.ylabel("Count")
plt.show()

# t-SNE on area and intensity
X = np.stack([areas, 10 * np.log10(means + 1e-6)], axis=1)
emb = TSNE(n_components=2, perplexity=25, random_state=42).fit_transform(X)
plt.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.6)
plt.title("t-SNE of Vehicle Patches")
plt.show()

print("Vehicle-level EDA complete. Saved summary to memory.")
