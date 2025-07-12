"""
Enhanced functions for analysis and visualization of SAR metadata from SICD XML files
Using only: numpy, matplotlib, shapely, scikit-learn, tqdm, ipywidgets
"""

# Standard Library Imports
import os
import logging
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

# External Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from ipywidgets import interact, IntSlider, Dropdown, Output
from IPython.display import display, HTML

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

class EnhancedSARMetadataAnalyzer:
    """
    Enhanced analyzer for SAR metadata with comprehensive visualization and analysis capabilities.
    """

    def __init__(self, metadata_extractor):
        """
        Initialize the Enhanced SAR Metadata Analyzer.

        Args:
            metadata_extractor: Instance of EnhancedSARMetadataExtractor
        """
        self.metadata_extractor = metadata_extractor
        self.metadata = metadata_extractor.metadata
        self.numeric_data, self.numeric_columns = metadata_extractor.get_numeric_data()
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Color palette

    def print_sample_metadata(self, n=5) -> None:
        """
        Print sample metadata entries.
        
        Args:
            n (int): Number of samples to display
        """
        if not isinstance(self.metadata, dict) or not self.metadata:
            print("No metadata available to display.")
            return
            
        sample = list(self.metadata.items())[:n]
        print(f"\nDisplaying {len(sample)} metadata entries:\n")
        for i, (filename, entry) in enumerate(sample, start=1):
            print(f"{i}. Filename: {filename}")
            for key, value in entry.items():
                if not isinstance(value, (list, dict)):  # Skip complex objects
                    print(f"   {key}: {value}")
            print()

    def generate_summary_statistics(self):
        """Generate and display comprehensive summary statistics."""
        if not self.metadata:
            print("No metadata loaded. Please load data first.")
            return
        
        stats = self.metadata_extractor.get_summary_statistics()
        
        print("=== Enhanced SICD Metadata Summary Statistics ===\n")
        
        # Basic info
        print(f"Total number of files: {stats['total_files']}")
        print(f"Numeric columns available: {len(self.numeric_columns) if self.numeric_columns else 0}")
        
        # Collectors
        if stats['collectors']:
            print("\nCollector Distribution:")
            total_files = stats['total_files']
            for collector, count in stats['collectors'].items():
                percentage = (count / total_files) * 100
                print(f"  {collector}: {count} files ({percentage:.1f}%)")
        
        # Radar Modes
        if stats['radar_modes']:
            print("\nRadar Mode Distribution:")
            for mode, count in stats['radar_modes'].items():
                percentage = (count / total_files) * 100
                print(f"  {mode}: {count} files ({percentage:.1f}%)")
        
        # Image characteristics
        if stats['image_statistics']:
            print("\nImage Characteristics:")
            for attr, attr_stats in stats['image_statistics'].items():
                print(f"  {attr}:")
                print(f"    Mean: {attr_stats['mean']:.2f}")
                print(f"    Std: {attr_stats['std']:.2f}")
                print(f"    Range: {attr_stats['min']:.2f} to {attr_stats['max']:.2f}")
                print(f"    Valid samples: {attr_stats['count']}")
        
        # Geographic extent
        if stats['geographic_extent']:
            geo = stats['geographic_extent']
            print("\nGeographic Extent:")
            print(f"  Latitude range: {geo['lat_min']:.4f} to {geo['lat_max']:.4f}")
            print(f"  Longitude range: {geo['lon_min']:.4f} to {geo['lon_max']:.4f}")
            print(f"  Center point: ({geo['center_lat']:.4f}, {geo['center_lon']:.4f})")
        
        # Temporal analysis
        if stats['date_range']['start']:
            print("\nTemporal Coverage:")
            print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    def create_overview_dashboard(self):
        """Create a comprehensive overview dashboard using matplotlib subplots."""
        if not self.metadata:
            print("No metadata loaded. Please load data first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SICD Metadata Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Collector distribution (pie chart)
        collectors = {}
        for metadata in self.metadata.values():
            collector = metadata.get('CollectorName', 'Unknown')
            collectors[collector] = collectors.get(collector, 0) + 1
        
        if collectors:
            axes[0, 0].pie(collectors.values(), labels=collectors.keys(), autopct='%1.1f%%', 
                          colors=self.colors[:len(collectors)])
            axes[0, 0].set_title('Collector Distribution')
        
        # 2. Radar mode distribution (bar chart)
        modes = {}
        for metadata in self.metadata.values():
            mode = metadata.get('ModeType', 'Unknown')
            modes[mode] = modes.get(mode, 0) + 1
        
        if modes:
            axes[0, 1].bar(modes.keys(), modes.values(), color=self.colors[:len(modes)])
            axes[0, 1].set_title('Radar Mode Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Image size distribution
        image_sizes = []
        for metadata in self.metadata.values():
            if 'ImageSize_MP' in metadata and metadata['ImageSize_MP'] is not None:
                image_sizes.append(metadata['ImageSize_MP'])
        
        if image_sizes:
            axes[0, 2].hist(image_sizes, bins=min(15, len(image_sizes)), alpha=0.7, color='skyblue')
            axes[0, 2].set_title('Image Size Distribution (MP)')
            axes[0, 2].set_xlabel('Megapixels')
            axes[0, 2].set_ylabel('Count')
        
        # 4. Geographic distribution
        lats = []
        lons = []
        for metadata in self.metadata.values():
            if metadata.get('SCP_Lat') is not None and metadata.get('SCP_Lon') is not None:
                lats.append(metadata['SCP_Lat'])
                lons.append(metadata['SCP_Lon'])
        
        if lats and lons:
            axes[1, 0].scatter(lons, lats, alpha=0.6, c='red', s=50)
            axes[1, 0].set_title('Geographic Distribution')
            axes[1, 0].set_xlabel('Longitude')
            axes[1, 0].set_ylabel('Latitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Incidence angle distribution
        incidence_angles = []
        for metadata in self.metadata.values():
            if 'IncidenceAng' in metadata and metadata['IncidenceAng'] is not None:
                incidence_angles.append(metadata['IncidenceAng'])
        
        if incidence_angles:
            axes[1, 1].hist(incidence_angles, bins=min(15, len(incidence_angles)), 
                           alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('Incidence Angle Distribution')
            axes[1, 1].set_xlabel('Angle (degrees)')
            axes[1, 1].set_ylabel('Count')
        
        # 6. Data completeness heatmap
        if self.metadata_extractor.column_names:
            completeness = []
            for col in self.metadata_extractor.column_names:
                count = 0
                for metadata in self.metadata.values():
                    if col in metadata and metadata[col] is not None:
                        count += 1
                completeness.append(count / len(self.metadata) * 100)
            
            # Show top 10 most complete columns
            sorted_indices = np.argsort(completeness)[-10:]
            top_cols = [self.metadata_extractor.column_names[i] for i in sorted_indices]
            top_completeness = [completeness[i] for i in sorted_indices]
            
            axes[1, 2].barh(top_cols, top_completeness, color='orange', alpha=0.7)
            axes[1, 2].set_title('Data Completeness (Top 10)')
            axes[1, 2].set_xlabel('Completeness (%)')
        
        plt.tight_layout()
        plt.show()

    def count_image_types(self) -> dict:
        """Count image types based on filename prefixes."""
        image_types = {}
        for filename in self.metadata.keys():
            if isinstance(filename, str):
                image_type = filename.split('_')[0]
                image_types[image_type] = image_types.get(image_type, 0) + 1
        return image_types

    def plot_image_type_counts(self, image_types: dict = None) -> None:
        """Plot image type counts."""
        if image_types is None:
            image_types = self.count_image_types()
        
        if not image_types:
            print("No image type data available for plotting.")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(image_types.keys(), image_types.values(), 
                      color=self.colors[:len(image_types)], edgecolor='black', alpha=0.8)
        plt.xlabel('Image Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Image Type Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def plot_image_count_by_mode(self, metadata: dict = None) -> None:
        """Plot image count by radar mode."""
        if metadata is None:
            metadata = self.metadata
        
        if not metadata:
            print("No metadata available for plotting.")
            return
        
        mode_counts = Counter(entry.get('ModeType', "Unknown") for entry in metadata.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(mode_counts.keys(), mode_counts.values(), 
                      color=self.colors[:len(mode_counts)], edgecolor='black', alpha=0.8)
        plt.xlabel("Radar Mode", fontsize=12)
        plt.ylabel("Image Count", fontsize=12)
        plt.title("Image Count by Radar Mode", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def plot_incidence_angle_distributions(self, metadata: dict = None, mode_type: str = None) -> None:
        """Plot incidence angle distributions."""
        if metadata is None:
            metadata = self.metadata
            
        if not metadata:
            print("No metadata available for plotting.")
            return
        
        # Filter angles by mode if specified
        if mode_type:
            angles = [entry["IncidenceAng"] for entry in metadata.values()
                      if entry.get("ModeType") == mode_type and "IncidenceAng" in entry 
                      and entry["IncidenceAng"] is not None]
            title = f"{mode_type} Incidence Angle Distribution"
        else:
            angles = [entry["IncidenceAng"] for entry in metadata.values()
                      if "IncidenceAng" in entry and entry["IncidenceAng"] is not None]
            title = "Incidence Angle Distribution (All Modes)"
        
        if angles:
            plt.figure(figsize=(10, 6))
            n, bins, patches = plt.hist(angles, bins=15, color='lightgreen', 
                                       edgecolor='darkgreen', alpha=0.7)
            plt.xlabel("Incidence Angle (degrees)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            plt.axvline(mean_angle, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_angle:.1f}°')
            plt.axvline(mean_angle + std_angle, color='orange', linestyle=':', 
                       alpha=0.7, label=f'±1σ: {std_angle:.1f}°')
            plt.axvline(mean_angle - std_angle, color='orange', linestyle=':', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"No incidence angle data found for {mode_type if mode_type else 'any mode'}.")

    def plot_metadata_distribution(self, metadata: dict = None, key: str = 'NumRows', title: str = None) -> None:
        """Plot distribution of a specific metadata attribute."""
        if metadata is None:
            metadata = self.metadata
            
        if not metadata:
            print("No metadata available for plotting.")
            return
        
        values = [float(entry.get(key)) for entry in metadata.values() 
                 if key in entry and entry[key] is not None]
        
        if values:
            plt.figure(figsize=(10, 6))
            n, bins, patches = plt.hist(values, bins=20, alpha=0.75, color='purple', 
                                       edgecolor='black')
            plt.title(title or f'{key} Distribution', fontsize=14, fontweight='bold')
            plt.xlabel(key, fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_val:.2f}')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data found for key '{key}'.")

    def plot_geospatial_data(self, metadata: dict = None, lat_min: float = None, lat_max: float = None,
                             lon_min: float = None, lon_max: float = None) -> None:
        """Plot geospatial distribution of SAR images."""
        if metadata is None:
            metadata = self.metadata
            
        if not metadata:
            print("No metadata available for plotting.")
            return
        
        # Set default bounds if not provided
        if lat_min is None:
            lat_min, lat_max = -90, 90
            lon_min, lon_max = -180, 180
        else:
            if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
                raise ValueError("Latitude values must be between -90 and 90 degrees")
            if not (-180 <= lon_min <= 180) or not (-180 <= lon_max <= 180):
                raise ValueError("Longitude values must be between -180 and 180 degrees")
            if lat_min > lat_max:
                raise ValueError("lat_min cannot be greater than lat_max")
            if lon_min > lon_max:
                raise ValueError("lon_min cannot be greater than lon_max")

        valid_entries = [entry for entry in metadata.values() 
                        if 'Lat' in entry and 'Lon' in entry 
                        and entry['Lat'] is not None and entry['Lon'] is not None]
        
        if not valid_entries:
            print("No valid coordinate data found in metadata.")
            return

        filtered_points = [
            (float(entry['Lon']), float(entry['Lat']), entry.get('Filename', 'Unknown'))
            for entry in valid_entries
            if lat_min <= float(entry['Lat']) <= lat_max and lon_min <= float(entry['Lon']) <= lon_max
        ]

        if not filtered_points:
            print("No points found within specified AOI.")
            return

        lons, lats, filenames = zip(*filtered_points)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(lons, lats, marker='o', color='blue', s=50, alpha=0.6, 
                             edgecolors='navy')
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.title("Geospatial Distribution of SICD Images", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add statistics
        plt.text(0.02, 0.98, f'Total points: {len(filtered_points)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def analyze_temporal_patterns(self):
        """Analyze and visualize temporal patterns in the data."""
        datetime_values = []
        for metadata in self.metadata.values():
            if 'DateTime' in metadata and metadata['DateTime']:
                try:
                    # Parse datetime string
                    dt_str = metadata['DateTime'].replace('Z', '')
                    if 'T' in dt_str:
                        dt = datetime.fromisoformat(dt_str)
                        datetime_values.append(dt)
                except:
                    pass
        
        if not datetime_values:
            print("No valid temporal data found for analysis.")
            return
        
        # Extract temporal components
        years = [dt.year for dt in datetime_values]
        months = [dt.month for dt in datetime_values]
        days_of_week = [dt.weekday() for dt in datetime_values]
        hours = [dt.hour for dt in datetime_values]
        
        # Create temporal analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Monthly distribution
        if len(set(months)) > 1:
            month_counts = Counter(months)
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[0, 0].bar([month_names[i-1] for i in sorted(month_counts.keys())], 
                          [month_counts[i] for i in sorted(month_counts.keys())], 
                          color='skyblue', edgecolor='navy')
            axes[0, 0].set_title('Acquisitions by Month')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Number of Images')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Day of week distribution
        if len(set(days_of_week)) > 1:
            dow_counts = Counter(days_of_week)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[0, 1].bar([day_names[i] for i in sorted(dow_counts.keys())], 
                          [dow_counts[i] for i in sorted(dow_counts.keys())], 
                          color='lightgreen', edgecolor='darkgreen')
            axes[0, 1].set_title('Acquisitions by Day of Week')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Number of Images')
        
        # Hourly distribution
        if len(set(hours)) > 1:
            hour_counts = Counter(hours)
            axes[1, 0].bar(sorted(hour_counts.keys()), 
                          [hour_counts[i] for i in sorted(hour_counts.keys())], 
                          color='orange', edgecolor='darkorange')
            axes[1, 0].set_title('Acquisitions by Hour of Day')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Number of Images')
            axes[1, 0].set_xticks(range(0, 24, 2))
        
        # Timeline scatter plot
        sorted_times = sorted(datetime_values)
        axes[1, 1].scatter(sorted_times, range(len(sorted_times)), alpha=0.6, color='purple')
        axes[1, 1].set_title('Acquisition Timeline')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Image Index')
        
        # Format x-axis dates
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print temporal statistics
        print("Temporal Statistics:")
        print(f"  Date range: {min(datetime_values)} to {max(datetime_values)}")
        print(f"  Total time span: {(max(datetime_values) - min(datetime_values)).days} days")
        print(f"  Total acquisitions: {len(datetime_values)}")

    def create_correlation_analysis(self):
        """Create correlation analysis for numeric variables."""
        if self.numeric_data is None or len(self.numeric_columns) < 2:
            print("Not enough numeric data for correlation analysis.")
            return
        
        # Remove NaN values for correlation calculation
        clean_data = []
        clean_columns = []
        
        for i, col in enumerate(self.numeric_columns):
            col_data = self.numeric_data[:, i]
            if not np.all(np.isnan(col_data)):
                clean_data.append(col_data)
                clean_columns.append(col)
        
        if len(clean_data) < 2:
            print("Not enough clean numeric data for correlation analysis.")
            return
        
        # Create correlation matrix
        clean_array = np.column_stack(clean_data)
        # Handle NaN values by using only complete cases
        complete_cases = ~np.any(np.isnan(clean_array), axis=1)
        if np.sum(complete_cases) < 2:
            print("Not enough complete cases for correlation analysis.")
            return
        
        clean_array = clean_array[complete_cases]
        corr_matrix = np.corrcoef(clean_array.T)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(range(len(clean_columns)))
        ax.set_yticks(range(len(clean_columns)))
        ax.set_xticklabels(clean_columns, rotation=45, ha='right')
        ax.set_yticklabels(clean_columns)
        
        # Add correlation values to cells
        for i in range(len(clean_columns)):
            for j in range(len(clean_columns)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Correlation Matrix of Numeric Metadata Variables', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        strong_corrs = []
        for i in range(len(clean_columns)):
            for j in range(i+1, len(clean_columns)):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append((clean_columns[i], clean_columns[j], corr_val))
        
        if strong_corrs:
            for var1, var2, corr_val in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {var1} vs {var2}: {corr_val:.3f}")
        else:
            print("  No strong correlations found.")

    def create_pca_analysis(self):
        """Perform PCA analysis on numeric data."""
        if self.numeric_data is None or len(self.numeric_columns) < 2:
            print("Not enough numeric data for PCA analysis.")
            return
        
        # Prepare data
        clean_data = []
        clean_columns = []
        
        for i, col in enumerate(self.numeric_columns):
            col_data = self.numeric_data[:, i]
            if not np.all(np.isnan(col_data)):
                clean_data.append(col_data)
                clean_columns.append(col)
        
        if len(clean_data) < 2:
            print("Not enough clean numeric data for PCA analysis.")
            return
        
        clean_array = np.column_stack(clean_data)
        complete_cases = ~np.any(np.isnan(clean_array), axis=1)
        
        if np.sum(complete_cases) < 3:
            print("Not enough complete cases for PCA analysis.")
            return
        
        clean_array = clean_array[complete_cases]
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_array)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance plot
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA: Explained Variance by Component')
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot of first two components
        if pca_result.shape[1] >= 2:
            axes[1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, color='red')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            axes[1].set_title('PCA: First Two Principal Components')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print component loadings
        print("PCA Component Loadings (first 3 components):")
        n_components = min(3, len(clean_columns))
        for i in range(n_components):
            print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]:.1%} of variance):")
            loadings = pca.components_[i]
            for j, col in enumerate(clean_columns):
                print(f"  {col}: {loadings[j]:.3f}")

    def create_clustering_analysis(self):
        """Perform clustering analysis on the metadata."""
        if self.numeric_data is None or len(self.numeric_columns) < 2:
            print("Not enough numeric data for clustering analysis.")
            return
        
        # Prepare data (same as PCA)
        clean_data = []
        clean_columns = []
        
        for i, col in enumerate(self.numeric_columns):
            col_data = self.numeric_data[:, i]
            if not np.all(np.isnan(col_data)):
                clean_data.append(col_data)
                clean_columns.append(col)
        
        if len(clean_data) < 2:
            print("Not enough clean numeric data for clustering analysis.")
            return
        
        clean_array = np.column_stack(clean_data)
        complete_cases = ~np.any(np.isnan(clean_array), axis=1)
        
        if np.sum(complete_cases) < 3:
            print("Not enough complete cases for clustering analysis.")
            return
        
        clean_array = clean_array[complete_cases]
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_array)
        
        # Determine optimal number of clusters using elbow method
        max_clusters = min(8, len(scaled_data) - 1)
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].grid(True, alpha=0.3)
        
        # Perform clustering with optimal k (choose k=3 as default)
        optimal_k = 3 if len(k_range) >= 3 else max(k_range)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Plot clustering results (first two features)
        scatter = axes[1].scatter(scaled_data[:, 0], scaled_data[:, 1], 
                                 c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
        axes[1].set_xlabel(f'{clean_columns[0]} (standardized)')
        axes[1].set_ylabel(f'{clean_columns[1]} (standardized)')
        axes[1].set_title(f'K-Means Clustering (k={optimal_k})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster statistics
        print(f"Clustering Results (k={optimal_k}):")
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            print(f"  Cluster {i}: {cluster_size} samples ({cluster_size/len(cluster_labels)*100:.1f}%)")

    def create_interactive_widgets(self):
        """Create interactive widgets for data exploration."""
        if not self.metadata:
            print("No metadata available for interactive analysis.")
            return
        
        output = Output()
        
        def plot_attribute_distribution(attribute):
            with output:
                output.clear_output(wait=True)
                self.plot_metadata_distribution(key=attribute, title=f'{attribute} Distribution')
        
        # Get available numeric attributes
        numeric_attrs = []
        for col in self.metadata_extractor.column_names:
            sample_values = [m.get(col) for m in self.metadata.values() if m.get(col) is not None]
            if sample_values and all(isinstance(v, (int, float)) for v in sample_values[:5]):
                numeric_attrs.append(col)
        
        if numeric_attrs:
            attribute_widget = Dropdown(
                options=numeric_attrs,
                value=numeric_attrs[0],
                description='Attribute:',
                disabled=False,
            )
            
            interact(plot_attribute_distribution, attribute=attribute_widget)
            display(output)
        else:
            print("No numeric attributes available for interactive analysis.")

    def export_summary_report(self, output_path: str = "enhanced_sicd_metadata_report.html"):
        """Export comprehensive summary report as HTML."""
        if not self.metadata:
            print("No metadata loaded. Please load data first.")
            return
        
        stats = self.metadata_extractor.get_summary_statistics()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced SICD Metadata Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background-color: #f8f9fa; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #667eea; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Enhanced SICD Metadata Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Analysis of SAR imagery metadata from SICD XML files</p>
                </div>
                
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_files']}</div>
                        <div class="stat-label">Total Files</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(self.metadata_extractor.column_names) if self.metadata_extractor.column_names else 'N/A'}</div>
                        <div class="stat-label">Attributes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(stats['collectors'])}</div>
                        <div class="stat-label">Unique Collectors</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(stats['radar_modes'])}</div>
                        <div class="stat-label">Radar Modes</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Collector Distribution</h2>
                    <table>
                        <tr><th>Collector</th><th>Count</th><th>Percentage</th></tr>
        """
        
        # Add collector data
        total_files = stats['total_files']
        for collector, count in stats['collectors'].items():
            percentage = (count / total_files) * 100
            html_content += f"<tr><td>{collector}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Radar Mode Distribution</h2>
                    <table>
                        <tr><th>Mode</th><th>Count</th><th>Percentage</th></tr>
        """
        
        # Add radar mode data
        for mode, count in stats['radar_modes'].items():
            percentage = (count / total_files) * 100
            html_content += f"<tr><td>{mode}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Geographic Coverage</h2>
        """
        
        if stats['geographic_extent']:
            geo = stats['geographic_extent']
            html_content += f"""
                    <p><strong>Latitude range:</strong> {geo['lat_min']:.4f} to {geo['lat_max']:.4f}</p>
                    <p><strong>Longitude range:</strong> {geo['lon_min']:.4f} to {geo['lon_max']:.4f}</p>
                    <p><strong>Center point:</strong> ({geo['center_lat']:.4f}, {geo['center_lon']:.4f})</p>
            """
        else:
            html_content += "<p>No geographic information available</p>"
        
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Temporal Coverage</h2>
        """
        
        if stats['date_range']['start']:
            html_content += f"<p><strong>Date range:</strong> {stats['date_range']['start']} to {stats['date_range']['end']}</p>"
        else:
            html_content += "<p>No temporal information available</p>"
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Enhanced report exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting report: {e}")