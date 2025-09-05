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

class DetectionReviewer:
    def __init__(self, nitf_path, json_report_path, class_map, excel_path):
        self.nitf_path = nitf_path
        self.json_report_path = json_report_path
        self.class_map = class_map
        self.excel_path = excel_path
        self.current_image_stem = None
        self.observations = []
        self.fig = None
        self.ax = None
        self.current_detections = {}
        
        # Load existing observations if Excel file exists
        self.load_existing_observations()
        
        # Initialize UI components
        self.setup_ui()
    
    def load_existing_observations(self):
        """Load existing observations from Excel file"""
        if os.path.exists(self.excel_path):
            try:
                df = pd.read_excel(self.excel_path)
                self.observations = df.to_dict('records')
                print(f"Loaded {len(self.observations)} existing observations from {self.excel_path}")
            except Exception as e:
                print(f"Could not load existing Excel file: {e}")
                self.observations = []
        else:
            self.observations = []
    
    def setup_ui(self):
        """Setup interactive UI widgets"""
        # Observation type dropdown
        self.obs_type = widgets.Dropdown(
            options=['False Positive', 'False Negative', 'True Positive', 'Missed Detection', 'Other'],
            description='Type:',
            style={'description_width': 'initial'}
        )
        
        # Error category dropdown
        self.error_category = widgets.Dropdown(
            options=['Localization Error', 'Classification Error', 'Background Confusion', 
                    'Occlusion', 'Scale Issue', 'Lighting/Quality', 'Annotation Error', 'Other'],
            description='Error Category:',
            style={'description_width': 'initial'}
        )
        
        # Severity rating
        self.severity = widgets.Dropdown(
            options=['Low', 'Medium', 'High', 'Critical'],
            description='Severity:',
            style={'description_width': 'initial'}
        )
        
        # Notes text area
        self.notes = widgets.Textarea(
            placeholder='Enter detailed observations...',
            description='Notes:',
            rows=3,
            style={'description_width': 'initial'}
        )
        
        # Buttons
        self.save_btn = widgets.Button(description='Save Observation', button_style='success')
        self.export_btn = widgets.Button(description='Export to Excel', button_style='info')
        self.stats_btn = widgets.Button(description='Show Statistics', button_style='warning')
        
        # Bind button events
        self.save_btn.on_click(self.save_observation)
        self.export_btn.on_click(self.export_to_excel)
        self.stats_btn.on_click(self.show_statistics)
    
    def create_interactive_plot(self, figsize=(16, 12)):
        """Create the main interactive visualization"""
        # Load JSON report
        with open(self.json_report_path, 'r') as f:
            report_json = json.load(f)
        
        # Process NITF image
        self.current_image_stem = Path(self.nitf_path).stem
        print(f"Loading image: {self.current_image_stem}")
        
        # Read and process image
        sicd_reader = SICDReader(self.nitf_path)
        original_row_ss = sicd_reader.sicd_meta.Grid.Row.SS
        original_col_ss = sicd_reader.sicd_meta.Grid.Col.SS
        
        height_scale_factor = original_row_ss / TARGET_SPACING
        width_scale_factor = original_col_ss / TARGET_SPACING
        
        original_height, original_width = sicd_reader.sicd_meta.ImageData.NumRows, sicd_reader.sicd_meta.ImageData.NumCols
        new_height = int(original_height * height_scale_factor)
        new_width = int(original_width * width_scale_factor)
        
        full_res_data = sicd_reader[:]
        remapped_full_res_img = Density()(full_res_data)
        del full_res_data
        
        img = cv2.resize(remapped_full_res_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        del remapped_full_res_img
        
        # Create plot
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.imshow(img, cmap='gray')
        
        # Plot detections with enhanced styling for review
        self.plot_detections_for_review(report_json, width_scale_factor, height_scale_factor)
        
        # Set up click handler for detailed inspection
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.ax.set_title(f"Detection Review: {self.current_image_stem}\\n"
                         f"Click on detections for details. Use toolbar to zoom/pan.\\n"
                         f"🟢 True Positives | 🔴 False Positives | 🟡 False Negatives | 🔵 Confused", 
                         fontsize=14, pad=20)
        self.ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        return self.fig
    
    def plot_detections_for_review(self, report_data, width_scale_factor, height_scale_factor):
        """Enhanced plotting for model review with clear FP/FN indicators"""
        
        # Enhanced styles for review
        styles = {
            'matched_detection': {'color': 'lime', 'linestyle': '-', 'linewidth': 2, 'label_prefix': '✓ TP'},
            'unmatched_detection': {'color': 'red', 'linestyle': '-', 'linewidth': 3, 'label_prefix': '✗ FP'},
            'unmatched_annotation': {'color': 'yellow', 'linestyle': '--', 'linewidth': 3, 'label_prefix': '✗ FN'},
            'confused_detection': {'color': 'aqua', 'linestyle': '-', 'linewidth': 2, 'label_prefix': '? CONF'},
            'confused_annotation': {'color': 'magenta', 'linestyle': '--', 'linewidth': 2, 'label_prefix': '? TRUE'}
        }
        
        detection_id = 0
        self.current_detections = {}
        
        def draw_detection(item_data, item_type, det_id):
            style = styles[item_type]
            x_min, y_min, x_max, y_max = item_data['bbox']
            
            plot_x = x_min * width_scale_factor
            plot_y = y_min * height_scale_factor
            plot_w = (x_max - x_min) * width_scale_factor
            plot_h = (y_max - y_min) * height_scale_factor
            
            # Store detection info for click handling
            self.current_detections[det_id] = {
                'bbox': [plot_x, plot_y, plot_x + plot_w, plot_y + plot_h],
                'original_bbox': item_data['bbox'],
                'type': item_type,
                'data': item_data,
                'style': style
            }
            
            # Draw bounding box
            patch = patches.Rectangle(
                (plot_x, plot_y), plot_w, plot_h,
                linewidth=style['linewidth'], 
                edgecolor=style['color'], 
                facecolor='none', 
                linestyle=style['linestyle'],
                alpha=0.8
            )
            self.ax.add_patch(patch)
            
            # Enhanced labels
            try:
                class_id = int(item_data['label'])
                display_label = self.class_map.get(class_id, f"ID:{class_id}")
            except (ValueError, TypeError):
                display_label = str(item_data['label'])
            
            confidence = f" ({item_data['score']:.2f})" if 'score' in item_data else ""
            text = f"{style['label_prefix']} {display_label}{confidence}"
            
            # Add detection ID for easy reference
            text += f" [{det_id}]"
            
            self.ax.text(
                plot_x, plot_y - 8, text,
                color=style['color'], fontsize=10, weight='bold', va='bottom',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor=style['color'], 
                         boxstyle='round,pad=0.3', linewidth=1)
            )
            
            return det_id + 1
        
        # Process different detection types
        keys_to_process = ['matched_detections', 'unmatched_detections', 'unmatched_annotations', 'confused_detections']
        
        for key in keys_to_process:
            if key not in report_data:
                continue
                
            items = report_data[key]
            
            if key == 'unmatched_detections':
                for item in items:
                    if Path(item['image']).stem == self.current_image_stem:
                        detection_id = draw_detection(item, 'unmatched_detection', detection_id)
            
            elif key == 'unmatched_annotations':
                for item in items:
                    if Path(item['image']).stem == self.current_image_stem:
                        detection_id = draw_detection(item, 'unmatched_annotation', detection_id)
            
            elif key in ['matched_detections', 'confused_detections']:
                for item in items:
                    if 'detect' in item and Path(item['detect']['image']).stem == self.current_image_stem:
                        det_type = 'confused_detection' if key == 'confused_detections' else 'matched_detection'
                        detection_id = draw_detection(item['detect'], det_type, detection_id)
                    
                    if 'annotation' in item and Path(item['annotation']['image']).stem == self.current_image_stem:
                        ann_type = 'confused_annotation' if key == 'confused_detections' else 'matched_annotation'
                        detection_id = draw_detection(item['annotation'], ann_type, detection_id)
        
        print(f"Loaded {len(self.current_detections)} detections for review")
        print("Legend: ✓ TP = True Positive, ✗ FP = False Positive, ✗ FN = False Negative, ? = Confused")
    
    def on_click(self, event):
        """Handle clicks on detections for detailed inspection"""
        if event.inaxes != self.ax:
            return
        
        click_x, click_y = event.xdata, event.ydata
        
        # Find clicked detection
        for det_id, det_info in self.current_detections.items():
            bbox = det_info['bbox']
            if bbox[0] <= click_x <= bbox[2] and bbox[1] <= click_y <= bbox[3]:
                self.show_detection_details(det_id, det_info)
                break
    
    def show_detection_details(self, det_id, det_info):
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
            class_name = self.class_map.get(int(class_id), f"Unknown ({class_id})")
            print(f"Class: {class_name} (ID: {class_id})")
        
        if 'score' in data:
            print(f"Confidence: {data['score']:.3f}")
        
        if 'image' in data:
            print(f"Image: {Path(data['image']).name}")
        
        print("-" * 60)
        
        # Pre-fill observation form
        detection_type = det_info['type']
        if 'unmatched_detection' in detection_type:
            self.obs_type.value = 'False Positive'
        elif 'unmatched_annotation' in detection_type:
            self.obs_type.value = 'False Negative'
        elif 'matched_detection' in detection_type:
            self.obs_type.value = 'True Positive'
        
        # Show observation form
        self.show_observation_form(det_id)
    
    def show_observation_form(self, detection_id=None):
        """Display the observation logging form"""
        print("\\n📝 LOG OBSERVATION:")
        
        # Set detection ID in notes if provided
        if detection_id is not None:
            current_notes = self.notes.value
            if f"Detection ID: {detection_id}" not in current_notes:
                self.notes.value = f"Detection ID: {detection_id}\\n{current_notes}"
        
        display(widgets.VBox([
            widgets.HTML("<b>Record your observation:</b>"),
            self.obs_type,
            self.error_category,
            self.severity,
            self.notes,
            widgets.HBox([self.save_btn, self.export_btn, self.stats_btn])
        ]))
    
    def save_observation(self, button):
        """Save current observation"""
        obs = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image': self.current_image_stem,
            'observation_type': self.obs_type.value,
            'error_category': self.error_category.value,
            'severity': self.severity.value,
            'notes': self.notes.value,
            'reviewer': os.getenv('USER', 'unknown')  # Get username
        }
        
        self.observations.append(obs)
        print(f"✅ Observation saved! Total observations: {len(self.observations)}")
        
        # Clear the form
        self.notes.value = ""
        
        # Auto-export to Excel
        self.export_to_excel(None)
    
    def export_to_excel(self, button):
        """Export all observations to Excel"""
        if not self.observations:
            print("No observations to export!")
            return
        
        df = pd.DataFrame(self.observations)
        
        # Create a detailed Excel file with multiple sheets
        with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
            # Main observations sheet
            df.to_excel(writer, sheet_name='Observations', index=False)
            
            # Summary statistics
            summary_stats = self.generate_summary_stats(df)
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            # Error breakdown
            error_breakdown = df.groupby(['observation_type', 'error_category']).size().reset_index(name='count')
            error_breakdown.to_excel(writer, sheet_name='Error_Breakdown', index=False)
        
        print(f"📊 Exported {len(self.observations)} observations to {self.excel_path}")
    
    def generate_summary_stats(self, df):
        """Generate summary statistics"""
        stats = []
        
        # Basic counts
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
        
        # By error category
        error_counts = df['error_category'].value_counts()
        for error, count in error_counts.head(5).items():  # Top 5
            stats.append({'Metric': f'Error: {error}', 'Value': count})
        
        return pd.DataFrame(stats)
    
    def show_statistics(self, button):
        """Display current statistics"""
        if not self.observations:
            print("No observations recorded yet!")
            return
        
        df = pd.DataFrame(self.observations)
        
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

# Convenience functions for notebook workflow
def setup_model_review():
    """Setup function to run at the beginning of review session"""
    try:
        get_ipython().run_line_magic('matplotlib', 'widget')
        print("✅ Interactive plotting enabled")
    except:
        try:
            get_ipython().run_line_magic('matplotlib', 'notebook')
            print("✅ Basic interactive plotting enabled")
        except:
            print("⚠️ Could not enable interactive plotting")
    
    print("\\n🔬 MODEL EVALUATION WORKFLOW")
    print("1. Update file paths below")
    print("2. Run: reviewer = start_review_session()")
    print("3. Use reviewer.create_interactive_plot() to begin")
    print("4. Click on detections to inspect and log observations")
    print("5. Use reviewer.show_statistics() to see progress")

def start_review_session(nitf_path=None, json_path=None, excel_path="review_log.xlsx"):
    """Start a new review session"""
    if not nitf_path or not json_path:
        print("❌ Please provide NITF and JSON paths:")
        print("reviewer = start_review_session('/path/to/image.nitf', '/path/to/report.json')")
        return None
    
    print(f"🚀 Starting review session...")
    print(f"   Image: {Path(nitf_path).name}")
    print(f"   Report: {Path(json_path).name}")
    print(f"   Log: {excel_path}")
    
    reviewer = DetectionReviewer(nitf_path, json_path, CLASS_MAP, excel_path)
    return reviewer

# Quick start instructions
print("🎯 QUICK START:")
print("1. Run: setup_model_review()")
print("2. Update paths and run: reviewer = start_review_session(nitf_path, json_path)")
print("3. Run: reviewer.create_interactive_plot()")
print("4. Start reviewing!")
