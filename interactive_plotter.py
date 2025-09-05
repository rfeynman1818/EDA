import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density
from pathlib import Path

TARGET_SPACING = 0.15  # meters

# Update these paths as needed
NITF_PATH = ""  # Path to your NITF file
JSON_REPORT_PATH = ""  # Path to your JSON report
KEYS_TO_VISUALIZE = ['matched_detections', 'unmatched_detections', 'unmatched_annotations', 'confused_detections']

CLASS_MAP = {
    0: "",  # Fill in your class names
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: ""
}

def plot_from_matching_report(ax, report_data, keys_to_plot, width_scale_factor, height_scale_factor, image_stem_to_match, class_map):
    """
    Plots bounding boxes from a report, using a class_map for display labels.
    """
    
    styles = {
        'matched_detection': {'color': 'lime', 'linestyle': '-', 'label_prefix': 'Matched'},
        'matched_annotation': {'color': 'lime', 'linestyle': '--', 'label_prefix': 'Matched'},
        'unmatched_detection': {'color': 'red', 'linestyle': '-', 'label_prefix': 'Unmatched Det'},
        'unmatched_annotation': {'color': 'yellow', 'linestyle': '--', 'label_prefix': 'Unmatched Anno'},
        'confused_detection': {'color': 'aqua', 'linestyle': '-', 'label_prefix': 'Confused Det'},
        'confused_annotation': {'color': 'magenta', 'linestyle': '--', 'label_prefix': 'True Label'}
    }
    
    plot_count = 0
    
    def _draw_box(item_data, item_type):
        """Helper function to draw a single, scaled bounding box with mapped labels."""
        style = styles[item_type]
        x_min, y_min, x_max, y_max = item_data['bbox']
        
        plot_x = x_min * width_scale_factor
        plot_y = y_min * height_scale_factor
        plot_w = (x_max - x_min) * width_scale_factor
        plot_h = (y_max - y_min) * height_scale_factor
        
        patch = patches.Rectangle(
            (plot_x, plot_y), plot_w, plot_h,
            linewidth=1.2, edgecolor=style['color'], facecolor='none', linestyle=style['linestyle']
        )
        ax.add_patch(patch)
        
        try:
            class_id = int(item_data['label'])
            display_label = class_map.get(class_id, f"Unknown ID: {class_id}")
        except (ValueError, TypeError):
            display_label = item_data['label']
        
        text = f"{style['label_prefix']}: {display_label}"
        if 'score' in item_data:
            text += f" ({item_data['score']:.2f})"
        
        ax.text(
            plot_x, plot_y - 5, text,
            color=style['color'], fontsize=8, va='bottom',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2')
        )
    
    for key in keys_to_plot:
        if key not in report_data:
            print(f"Warning: Key '{key}' not found in JSON report. Skipping.")
            continue
        
        print(f"Processing '{key}'...")
        items_to_plot = report_data[key]
        
        if key in ['unmatched_annotations', 'unmatched_detections']:
            item_type = 'unmatched_annotation' if key == 'unmatched_annotations' else 'unmatched_detection'
            for item in items_to_plot:
                if Path(item['image']).stem == image_stem_to_match:
                    _draw_box(item, item_type)
                    plot_count += 1
        
        elif key in ['matched_detections', 'confused_detections']:
            for item in items_to_plot:
                if 'detect' in item and Path(item['detect']['image']).stem == image_stem_to_match:
                    item_type = f"{key.replace('_detections', '')}_detection"
                    _draw_box(item['detect'], item_type)
                    plot_count += 1
                
                if 'annotation' in item and Path(item['annotation']['image']).stem == image_stem_to_match:
                    item_type = 'confused_annotation' if key == 'confused_detections' else f"{key.replace('_detections', '')}_annotation"
                    _draw_box(item['annotation'], item_type)
                
                if 'detect' not in item:
                    plot_count += 1
    
    print(f"Plotted {plot_count} matching annotations/detections for image '{image_stem_to_match}'.")

def create_interactive_visualization(nitf_path, json_report_path, keys_to_visualize, class_map, figsize=(15, 15)):
    """
    Create an interactive visualization that can be zoomed and panned in Jupyter notebook.
    
    Args:
        nitf_path: Path to NITF file
        json_report_path: Path to JSON report
        keys_to_visualize: List of keys to plot
        class_map: Dictionary mapping class IDs to names
        figsize: Figure size tuple
    
    Returns:
        matplotlib.figure.Figure: Interactive figure object
    """
    
    # Load JSON report
    with open(json_report_path, 'r') as f:
        report_json = json.load(f)
    
    # Process NITF image
    nitf_stem = Path(nitf_path).stem
    print(f"Targeting image stem: {nitf_stem}")
    
    print("Reading and preparing image...")
    sicd_reader = SICDReader(nitf_path)
    
    original_row_ss = sicd_reader.sicd_meta.Grid.Row.SS
    original_col_ss = sicd_reader.sicd_meta.Grid.Col.SS
    print(f"Original pixel spacing (Row, Col): ({original_row_ss:.4f}m, {original_col_ss:.4f}m)")
    print(f"Target pixel spacing (Row, Col): ({TARGET_SPACING:.4f}m, {TARGET_SPACING:.4f}m)")
    
    height_scale_factor = original_row_ss / TARGET_SPACING
    width_scale_factor = original_col_ss / TARGET_SPACING
    
    original_height, original_width = sicd_reader.sicd_meta.ImageData.NumRows, sicd_reader.sicd_meta.ImageData.NumCols
    new_height = int(original_height * height_scale_factor)
    new_width = int(original_width * width_scale_factor)
    
    full_res_data = sicd_reader[:]
    remapped_full_res_img = Density()(full_res_data)
    del full_res_data
    
    print(f"Resizing image from ({original_width}, {original_height}) to ({new_width}, {new_height})")
    img = cv2.resize(remapped_full_res_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    del remapped_full_res_img
    print(f"Display image shape: {img.shape}")
    
    # Create figure and axis with interactive features
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray')
    
    # Plot the annotations/detections
    plot_from_matching_report(ax, report_json, keys_to_visualize, width_scale_factor, height_scale_factor, nitf_stem, class_map)
    
    # Set title and formatting
    ax.set_title(f"Interactive Visualization: {', '.join(keys_to_visualize)}\\nImage: {nitf_stem} (Resampled to {TARGET_SPACING}m spacing)")
    ax.axis('off')
    
    # Enable interactive navigation
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return fig

def setup_jupyter_interactivity():
    """
    Helper function to set up interactive plotting in Jupyter.
    Run this in a cell before creating visualizations.
    """
    try:
        # Try to use widget backend for best interactivity
        get_ipython().run_line_magic('matplotlib', 'widget')
        print("Using interactive widget backend. You can zoom, pan, and interact with plots.")
        print("If you encounter issues, try running: %matplotlib notebook")
    except:
        try:
            # Fallback to notebook backend
            get_ipython().run_line_magic('matplotlib', 'notebook')
            print("Using notebook backend for basic interactivity.")
        except:
            print("Could not set up interactive backend. Make sure you're running in Jupyter.")
            print("Try manually running: %matplotlib widget or %matplotlib notebook")

# Example usage for Jupyter notebook:
def run_interactive_visualization():
    """
    Run the interactive visualization. Update the paths before calling.
    """
    # Update these paths to your actual files
    if not NITF_PATH or not JSON_REPORT_PATH:
        print("Please update NITF_PATH and JSON_REPORT_PATH variables before running.")
        print("Example:")
        print('NITF_PATH = "/path/to/your/file.nitf"')
        print('JSON_REPORT_PATH = "/path/to/your/report.json"')
        return
    
    # Create interactive visualization
    fig = create_interactive_visualization(
        nitf_path=NITF_PATH,
        json_report_path=JSON_REPORT_PATH,
        keys_to_visualize=KEYS_TO_VISUALIZE,
        class_map=CLASS_MAP,
        figsize=(15, 15)  # Adjust size as needed
    )
    
    return fig

