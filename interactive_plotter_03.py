import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
from IPython.display import display, clear_output
import ipywidgets as widgets

TARGET_SPACING = 0.15  # meters

# File paths - UPDATE THESE
NITF_PATH = ""  # Path to your NITF file
JSON_REPORT_PATH = ""  # Path to your JSON report
EXCEL_LOG_PATH = "detection_review_log.xlsx"  # Where to save observations

CLASS_MAP = {
    0: "background",
    1: "vehicle", 
    2: "aircraft",
    3: "ship",
    4: "building",
    5: "person",
    6: "other"
}

KEYS_TO_VISUALIZE = ['matched_detections', 'unmatched_detections', 'unmatched_annotations', 'confused_detections']

# Global variables for observation tracking
current_detections = {}
current_image_stem = None
observations = []

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

def plot_from_matching_report(ax, report_data, keys_to_plot, width_scale_factor, height_scale_factor, image_stem_to_match, class_map):
    """
    Plots bounding boxes from a report, using a class_map for display labels.
    Enhanced for model evaluation with clickable detections.
    """
    global current_detections
    current_detections = {}
    
    styles = {
        'matched_detection': {'color': 'lime', 'linestyle': '-', 'linewidth': 2, 'label_prefix': '✓ TP'},
        'matched_annotation': {'color': 'lime', 'linestyle': '--', 'linewidth': 2, 'label_prefix': '✓ GT'},
        'unmatched_detection': {'color': 'red', 'linestyle': '-', 'linewidth': 3, 'label_prefix': '✗ FP'},
        'unmatched_annotation': {'color': 'yellow', 'linestyle': '--', 'linewidth': 3, 'label_prefix': '✗ FN'},
        'confused_detection': {'color': 'aqua', 'linestyle': '-', 'linewidth': 2, 'label_prefix': '? CONF'},
        'confused_annotation': {'color': 'magenta', 'linestyle': '--', 'linewidth': 2, 'label_prefix': '? TRUE'}
    }
    
    plot_count = 0
    detection_id = 0
    
    def _draw_box(item_data, item_type, det_id):
        """Helper function to draw a single, scaled bounding box with mapped labels."""
        nonlocal plot_count
        
        style = styles.get(item_type, {'color': 'white', 'linestyle': '-', 'linewidth': 1, 'label_prefix': '?'})
        x_min, y_min, x_max, y_max = item_data['bbox']
        
        plot_x = x_min * width_scale_factor
        plot_y = y_min * height_scale_factor
        plot_w = (x_max - x_min) * width_scale_factor
        plot_h = (y_max - y_min) * height_scale_factor
        
        # Store detection info for click handling
        current_detections[det_id] = {
            'bbox': [plot_x, plot_y, plot_x + plot_w, plot_y + plot_h],
            'original_bbox': item_data['bbox'],
            'type': item_type,
            'data': item_data,
            'style': style
        }
        
        patch = patches.Rectangle(
            (plot_x, plot_y), plot_w, plot_h,
            linewidth=style['linewidth'], edgecolor=style['color'], facecolor='none', linestyle=style['linestyle']
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
        
        # Add detection ID for easy reference
        text += f" [ID:{det_id}]"
        
        ax.text(
            plot_x, plot_y - 8, text,
            color=style['color'], fontsize=10, weight='bold', va='bottom',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor=style['color'], 
                     boxstyle='round,pad=0.3', linewidth=1)
        )
        plot_count += 1
        return det_id + 1
    
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
                    detection_id = _draw_box(item, item_type, detection_id)
        
        elif key in ['matched_detections', 'confused_detections']:
            for item in items_to_plot:
                if 'detect' in item and Path(item['detect']['image']).stem == image_stem_to_match:
                    item_type = f"{key.replace('_detections', '')}_detection"
                    detection_id = _draw_box(item['detect'], item_type, detection_id)
                
                if 'annotation' in item and Path(item['annotation']['image']).stem == image_stem_to_match:
                    item_type = 'confused_annotation' if key == 'confused_detections' else f"{key.replace('_detections', '')}_annotation"
                    detection_id = _draw_box(item['annotation'], item_type, detection_id)
    
    print(f"Plotted {plot_count} matching annotations/detections for image '{image_stem_to_match}'.")
    print("Legend: ✓ TP = True Positive, ✗ FP = False Positive, ✗ FN = False Negative, ? = Confused")
    print("Click on any detection box to inspect details and log observations.")

def create_interactive_visualization(nitf_path, json_report_path, keys_to_visualize, class_map, figsize=(16, 12)):
    """
    Create an interactive visualization that can be zoomed and panned in Jupyter notebook.
    Enhanced for model evaluation with clickable detections.
    """
    global current_image_stem
    
    # Load JSON report
    with open(json_report_path, 'r') as f:
        report_json = json.load(f)
    
    # Process NITF image
    current_image_stem = Path(nitf_path).stem
    print(f"Targeting image stem: {current_image_stem}")
    
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
    
    # Plot the annotations/detections with enhanced styling
    plot_from_matching_report(ax, report_json, keys_to_visualize, width_scale_factor, height_scale_factor, current_image_stem, class_map)
    
    # Set up click handler for detailed inspection
    fig.canvas.mpl_connect('button_press_event', on_detection_click)
    
    # Set title and formatting
    ax.set_title(f"Model Evaluation: {', '.join(keys_to_visualize)}\\n"
                f"Image: {current_image_stem} (Resampled to {TARGET_SPACING}m spacing)\\n"
                f"Click on detections for details • Use toolbar to zoom/pan", 
                fontsize=14, pad=20)
    ax.axis('off')
    
    # Enable interactive navigation
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return fig

def on_detection_click(event):
    """Handle clicks on detections for detailed inspection"""
    if event.inaxes is None:
        return
    
    click_x, click_y = event.xdata, event.ydata
    if click_x is None or click_y is None:
        return
    
    # Find clicked detection
    for det_id, det_info in current_detections.items():
        bbox = det_info['bbox']
        if bbox[0] <= click_x <= bbox[2] and bbox[1] <= click_y <= bbox[3]:
            show_detection_details(det_id, det_info)
            break

def show_detection_details(det_id, det_info):
    """Show detailed information about clicked detection"""
    clear_output(wait=True)
    
    print("=" * 60)
    print(f"DETECTION DETAILS - ID: {det_id}")
    print("=" * 60)
    print(f"Type: {det_info['type'].replace('_', ' ').title()}")
    print(f"Bounding Box: {det_info['original_bbox']}")
    
    data = det_info['data']
    if 'label' in data:
        class_id = data['label']
        class_name = CLASS_MAP.get(int(class_id), f"Unknown ({class_id})")
        print(f"Class: {class_name} (ID: {class_id})")
    
    if 'score' in data:
        print(f"Confidence: {data['score']:.3f}")
    
    if 'image' in data:
        print(f"Image: {Path(data['image']).name}")
    
    print("-" * 60)
    
    # Show observation form
    show_observation_form(det_id, det_info['type'])

def show_observation_form(detection_id=None, detection_type=None):
    """Display the observation logging form"""
    print("\\n📝 LOG OBSERVATION:")
    
    # Create form widgets
    obs_type = widgets.Dropdown(
        options=['False Positive', 'False Negative', 'True Positive', 'Missed Detection', 'Other'],
        description='Type:',
        style={'description_width': 'initial'}
    )
    
    # Pre-fill based on detection type
    if detection_type:
        if 'unmatched_detection' in detection_type:
            obs_type.value = 'False Positive'
        elif 'unmatched_annotation' in detection_type:
            obs_type.value = 'False Negative'
        elif 'matched_detection' in detection_type:
            obs_type.value = 'True Positive'
    
    error_category = widgets.Dropdown(
        options=['Localization Error', 'Classification Error', 'Background Confusion', 
                'Occlusion', 'Scale Issue', 'Lighting/Quality', 'Annotation Error', 'Other'],
        description='Error Category:',
        style={'description_width': 'initial'}
    )
    
    severity = widgets.Dropdown(
        options=['Low', 'Medium', 'High', 'Critical'],
        description='Severity:',
        style={'description_width': 'initial'}
    )
    
    notes = widgets.Textarea(
        value=f"Detection ID: {detection_id}\\n" if detection_id else "",
        placeholder='Enter detailed observations...',
        description='Notes:',
        rows=3,
        style={'description_width': 'initial'}
    )
    
    def save_observation(button):
        obs = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image': current_image_stem,
            'detection_id': detection_id,
            'observation_type': obs_type.value,
            'error_category': error_category.value,
            'severity': severity.value,
            'notes': notes.value,
            'reviewer': os.getenv('USER', 'unknown')
        }
        
        observations.append(obs)
        print(f"✅ Observation saved! Total observations: {len(observations)}")
        
        # Auto-export to Excel
        export_observations_to_excel()
        
        # Clear form
        notes.value = f"Detection ID: {detection_id + 1}\\n" if detection_id else ""
    
    def show_stats(button):
        show_current_statistics()
    
    save_btn = widgets.Button(description='Save Observation', button_style='success')
    stats_btn = widgets.Button(description='Show Statistics', button_style='info')
    
    save_btn.on_click(save_observation)
    stats_btn.on_click(show_stats)
    
    display(widgets.VBox([
        widgets.HTML("<b>Record your observation:</b>"),
        obs_type,
        error_category,
        severity,
        notes,
        widgets.HBox([save_btn, stats_btn])
    ]))

def load_existing_observations():
    """Load existing observations from Excel file"""
    global observations
    if os.path.exists(EXCEL_LOG_PATH):
        try:
            df = pd.read_excel(EXCEL_LOG_PATH)
            observations = df.to_dict('records')
            print(f"Loaded {len(observations)} existing observations from {EXCEL_LOG_PATH}")
        except Exception as e:
            print(f"Could not load existing Excel file: {e}")
            observations = []
    else:
        observations = []

def export_observations_to_excel():
    """Export all observations to Excel"""
    if not observations:
        print("No observations to export!")
        return
    
    df = pd.DataFrame(observations)
    
    # Create a detailed Excel file with multiple sheets
    with pd.ExcelWriter(EXCEL_LOG_PATH, engine='openpyxl') as writer:
        # Main observations sheet
        df.to_excel(writer, sheet_name='Observations', index=False)
        
        # Summary statistics
        summary_stats = generate_summary_stats(df)
        summary_stats.to_excel(writer, sheet_name='Summary', index=False)
        
        # Error breakdown
        if len(df) > 0:
            error_breakdown = df.groupby(['observation_type', 'error_category']).size().reset_index(name='count')
            error_breakdown.to_excel(writer, sheet_name='Error_Breakdown', index=False)
    
    print(f"📊 Exported {len(observations)} observations to {EXCEL_LOG_PATH}")

def generate_summary_stats(df):
    """Generate summary statistics"""
    if len(df) == 0:
        return pd.DataFrame([{'Metric': 'No observations', 'Value': 0}])
    
    stats = []
    total_obs = len(df)
    stats.append({'Metric': 'Total Observations', 'Value': total_obs})
    
    # By type
    type_counts = df['observation_type'].value_counts()
    for obs_type, count in type_counts.items():
        stats.append({'Metric': f'{obs_type} Count', 'Value': count})
        stats.append({'Metric': f'{obs_type} %', 'Value': f"{(count/total_obs)*100:.1f}%"})
    
    # By severity
    severity_counts = df['severity'].value_counts()
    for severity, count in severity_counts.items():
        stats.append({'Metric': f'{severity} Severity', 'Value': count})
    
    return pd.DataFrame(stats)

def show_current_statistics():
    """Display current statistics"""
    if not observations:
        print("No observations recorded yet!")
        return
    
    df = pd.DataFrame(observations)
    
    print("\\n📊 CURRENT STATISTICS")
    print("=" * 50)
    print(f"Total Observations: {len(df)}")
    print(f"Images Reviewed: {df['image'].nunique()}")
    
    print("\\n📈 Observation Types:")
    type_counts = df['observation_type'].value_counts()
    for obs_type, count in type_counts.items():
        pct = (count/len(df))*100
        print(f"  {obs_type}: {count} ({pct:.1f}%)")
    
    print("\\n⚠️ Error Categories:")
    error_counts = df['error_category'].value_counts().head(5)
    for error, count in error_counts.items():
        print(f"  {error}: {count}")
    
    print("\\n🔥 Severity Distribution:")
    severity_counts = df['severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}")

# Main workflow functions
def start_model_evaluation():
    """Start the model evaluation workflow"""
    print("🔬 MODEL EVALUATION WORKFLOW")
    print("1. Run: setup_jupyter_interactivity()")
    print("2. Update file paths: NITF_PATH, JSON_REPORT_PATH, CLASS_MAP")
    print("3. Run: load_existing_observations()")
    print("4. Run: fig = create_review_plot()")
    print("5. Click on detections to inspect and log observations")
    print("6. Use show_current_statistics() to see progress")

def create_review_plot():
    """Create the main review plot"""
    if not NITF_PATH or not JSON_REPORT_PATH:
        print("❌ Please update NITF_PATH and JSON_REPORT_PATH variables first!")
        print("Example:")
        print('NITF_PATH = "/path/to/your/file.nitf"')
        print('JSON_REPORT_PATH = "/path/to/your/report.json"')
        return None
    
    print(f"🚀 Creating review plot...")
    print(f"   Image: {Path(NITF_PATH).name}")
    print(f"   Report: {Path(JSON_REPORT_PATH).name}")
    
    fig = create_interactive_visualization(
        nitf_path=NITF_PATH,
        json_report_path=JSON_REPORT_PATH,
        keys_to_visualize=KEYS_TO_VISUALIZE,
        class_map=CLASS_MAP,
        figsize=(16, 12)
    )
    
    return fig

# Initialize on import
print("🎯 DETECTION MODEL EVALUATION TOOLKIT")
print("Quick start: run start_model_evaluation() for instructions")
