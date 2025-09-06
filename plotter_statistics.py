import pandas as pd

def export_detections_to_csv(filename="all_detections.csv"):
    """Export all loaded detections to a CSV file"""
    
    if not current_detections:
        print("No detections loaded. Run create_review_plot() first.")
        return
    
    # Convert detections to list of dictionaries
    detection_records = []
    
    for det_id, det_info in current_detections.items():
        data = det_info['data']
        
        record = {
            'detection_id': det_id,
            'image': current_image_stem,
            'detection_type': det_info['type'],
            'bbox_x_min': det_info['original_bbox'][0],
            'bbox_y_min': det_info['original_bbox'][1], 
            'bbox_x_max': det_info['original_bbox'][2],
            'bbox_y_max': det_info['original_bbox'][3],
            'bbox_width': det_info['original_bbox'][2] - det_info['original_bbox'][0],
            'bbox_height': det_info['original_bbox'][3] - det_info['original_bbox'][1],
            'class_id': data.get('label', 'N/A'),
            'class_name': CLASS_MAP.get(int(data.get('label', 0)), 'unknown') if 'label' in data else 'unknown',
            'confidence_score': data.get('score', 'N/A'),
            'style_color': det_info['style']['color'],
            'label_prefix': det_info['style']['label_prefix']
        }
        
        # Add any other fields from the original data
        for key, value in data.items():
            if key not in ['bbox', 'label', 'score']:
                record[f'extra_{key}'] = value
                
        detection_records.append(record)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(detection_records)
    
    # Sort by detection_id for easier reading
    df = df.sort_values('detection_id')
    
    # Save to CSV
    csv_path = filename
    df.to_csv(csv_path, index=False)
    
    print(f"Exported {len(detection_records)} detections to: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Show summary statistics
    print(f"\nDetection Type Summary:")
    print(df['detection_type'].value_counts())
    
    print(f"\nClass Distribution:")
    print(df['class_name'].value_counts())
    
    return df

# Usage:
df = export_detections_to_csv("fuzzy_detections.csv")

# Optional: preview the data
print("\nFirst 5 rows:")
print(df.head())

###############################################################

def export_simple_detections_csv(filename="detections_simple.csv"):
    """Export a simplified CSV with just key detection info"""
    
    records = []
    for det_id, det_info in current_detections.items():
        data = det_info['data']
        records.append({
            'id': det_id,
            'type': det_info['type'],
            'class': CLASS_MAP.get(int(data.get('label', 0)), 'unknown'),
            'confidence': data.get('score', 'N/A'),
            'x_min': det_info['original_bbox'][0],
            'y_min': det_info['original_bbox'][1],
            'x_max': det_info['original_bbox'][2], 
            'y_max': det_info['original_bbox'][3]
        })
    
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Exported {len(records)} detections to {filename}")
    return df

# For a quick simple export:
simple_df = export_simple_detections_csv()


###############################################################

# Step 1: Run the export after your plot

# After running create_review_plot(), run this in any cell:
df = export_detections_to_csv("fuzzy_detections.csv")

###############################################################

# Step 2: Focus on False Positives

# Extract only false positives
fp_df = df[df['detection_type'] == 'unmatched_detection'].copy()
fp_df.to_csv("false_positives.csv", index=False)

print(f"Found {len(fp_df)} False Positives")
print("\nFP Class Distribution:")
print(fp_df['class_name'].value_counts())

print("\nFP Confidence Score Stats:")
print(fp_df['confidence_score'].describe())

# Show lowest confidence FPs (likely easiest to fix)
print("\nLowest confidence FPs:")
print(fp_df.nsmallest(10, 'confidence_score')[['detection_id', 'class_name', 'confidence_score']])

###############################################################

# Step 3: Focus on False Negatives

# Extract only false negatives  
fn_df = df[df['detection_type'] == 'unmatched_annotation'].copy()
fn_df.to_csv("false_negatives.csv", index=False)

print(f"Found {len(fn_df)} False Negatives")
print("\nFN Class Distribution:")
print(fn_df['class_name'].value_counts())

# Analyze FN bounding box sizes (small objects harder to detect)
fn_df['bbox_area'] = fn_df['bbox_width'] * fn_df['bbox_height']
print("\nFN Bounding Box Area Stats:")
print(fn_df['bbox_area'].describe())

print("\nSmallest missed objects:")
print(fn_df.nsmallest(10, 'bbox_area')[['detection_id', 'class_name', 'bbox_area']])

###############################################################

# Step 4: Compare FPs vs FNs

# Quick comparison
fp_count = len(df[df['detection_type'] == 'unmatched_detection'])
fn_count = len(df[df['detection_type'] == 'unmatched_annotation'])
tp_count = len(df[df['detection_type'] == 'matched_detection'])

print(f"Detection Performance Summary:")
print(f"True Positives: {tp_count}")
print(f"False Positives: {fp_count}")
print(f"False Negatives: {fn_count}")
print(f"Precision: {tp_count/(tp_count + fp_count):.3f}")
print(f"Recall: {tp_count/(tp_count + fn_count):.3f}")

###############################################################

# Step 5: Spatial Analysis (Optional)

# Check if errors cluster in certain image regions
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot FP locations
ax1.scatter(fp_df['bbox_x_min'], fp_df['bbox_y_min'], c='red', alpha=0.6, s=20)
ax1.set_title(f'False Positive Locations ({len(fp_df)} total)')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')

# Plot FN locations  
ax2.scatter(fn_df['bbox_x_min'], fn_df['bbox_y_min'], c='yellow', alpha=0.6, s=20)
ax2.set_title(f'False Negative Locations ({len(fn_df)} total)')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')

plt.tight_layout()
plt.show()

###############################################################

# 1. Which classes have the most errors

# Count errors by class
error_df = df[df['detection_type'].isin(['unmatched_detection', 'unmatched_annotation'])]

print("Errors by Class:")
class_errors = error_df['class_name'].value_counts()
print(class_errors)

# Show error breakdown by type
error_breakdown = df.groupby(['class_name', 'detection_type']).size().unstack(fill_value=0)
print("\nError Breakdown by Class:")
print(error_breakdown)

# Calculate error rates
tp_by_class = df[df['detection_type'] == 'matched_detection']['class_name'].value_counts()
total_by_class = df['class_name'].value_counts()

print("\nError Rate by Class:")
for class_name in total_by_class.index:
    total = total_by_class[class_name]
    errors = class_errors.get(class_name, 0)
    error_rate = errors / total * 100
    print(f"{class_name}: {errors}/{total} = {error_rate:.1f}% error rate")

###############################################################

# 2. Low confidence vs false positives correlation

# Analyze confidence scores for FPs vs TPs
tp_df = df[df['detection_type'] == 'matched_detection']
fp_df = df[df['detection_type'] == 'unmatched_detection']

print("Confidence Score Analysis:")
print(f"TP average confidence: {tp_df['confidence_score'].mean():.3f}")
print(f"FP average confidence: {fp_df['confidence_score'].mean():.3f}")

# Count low confidence detections
low_conf_threshold = 0.5
tp_low_conf = len(tp_df[tp_df['confidence_score'] < low_conf_threshold])
fp_low_conf = len(fp_df[fp_df['confidence_score'] < low_conf_threshold])

print(f"\nLow confidence (<{low_conf_threshold}) breakdown:")
print(f"TPs with low confidence: {tp_low_conf}/{len(tp_df)} = {tp_low_conf/len(tp_df)*100:.1f}%")
print(f"FPs with low confidence: {fp_low_conf}/{len(fp_df)} = {fp_low_conf/len(fp_df)*100:.1f}%")

# Show confidence distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(tp_df['confidence_score'], bins=20, alpha=0.6, label='True Positives', color='green')
plt.hist(fp_df['confidence_score'], bins=20, alpha=0.6, label='False Positives', color='red')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.title('Confidence Score Distribution: TP vs FP')
plt.legend()
plt.show()

###############################################################

# 3. Missed objects size analysis

# Analyze size of missed objects (FNs)
fn_df = df[df['detection_type'] == 'unmatched_annotation']
tp_df = df[df['detection_type'] == 'matched_detection']

# Calculate areas
fn_df = fn_df.copy()
fn_df['bbox_area'] = fn_df['bbox_width'] * fn_df['bbox_height']
tp_df = tp_df.copy()
tp_df['bbox_area'] = tp_df['bbox_width'] * tp_df['bbox_height']

print("Object Size Analysis:")
print(f"Average FN (missed) object area: {fn_df['bbox_area'].mean():.1f}")
print(f"Average TP (detected) object area: {tp_df['bbox_area'].mean():.1f}")

# Size percentiles
print(f"\nFN size percentiles:")
print(fn_df['bbox_area'].describe())

print(f"\nTP size percentiles:")
print(tp_df['bbox_area'].describe())

# Small object analysis
small_threshold = fn_df['bbox_area'].quantile(0.25)  # Bottom 25%
small_fn = len(fn_df[fn_df['bbox_area'] <= small_threshold])
small_tp = len(tp_df[tp_df['bbox_area'] <= small_threshold])

print(f"\nSmall objects (area <= {small_threshold:.1f}):")
print(f"Small FNs: {small_fn}/{len(fn_df)} = {small_fn/len(fn_df)*100:.1f}%")
print(f"Small TPs: {small_tp}/{len(tp_df)} = {small_tp/len(tp_df)*100:.1f}%")

###############################################################

# 4. Spatial clustering analysis

# Analyze spatial distribution of errors
import numpy as np

# Calculate image quadrants
x_mid = df['bbox_x_min'].median()
y_mid = df['bbox_y_min'].median()

def get_quadrant(row):
    if row['bbox_x_min'] <= x_mid and row['bbox_y_min'] <= y_mid:
        return 'Top-Left'
    elif row['bbox_x_min'] > x_mid and row['bbox_y_min'] <= y_mid:
        return 'Top-Right'
    elif row['bbox_x_min'] <= x_mid and row['bbox_y_min'] > y_mid:
        return 'Bottom-Left'
    else:
        return 'Bottom-Right'

df['quadrant'] = df.apply(get_quadrant, axis=1)

print("Spatial Distribution Analysis:")
print("\nErrors by Image Quadrant:")
error_by_quad = df[df['detection_type'].isin(['unmatched_detection', 'unmatched_annotation'])]['quadrant'].value_counts()
total_by_quad = df['quadrant'].value_counts()

for quad in total_by_quad.index:
    errors = error_by_quad.get(quad, 0)
    total = total_by_quad[quad]
    print(f"{quad}: {errors}/{total} = {errors/total*100:.1f}% errors")

# Distance from center analysis
center_x = df['bbox_x_min'].median()
center_y = df['bbox_y_min'].median()

df['distance_from_center'] = np.sqrt((df['bbox_x_min'] - center_x)**2 + (df['bbox_y_min'] - center_y)**2)

# Compare error rates by distance from center
df['distance_category'] = pd.cut(df['distance_from_center'], bins=3, labels=['Center', 'Middle', 'Edge'])

print(f"\nError rates by distance from image center:")
for dist_cat in ['Center', 'Middle', 'Edge']:
    subset = df[df['distance_category'] == dist_cat]
    errors = len(subset[subset['detection_type'].isin(['unmatched_detection', 'unmatched_annotation'])])
    total = len(subset)
    if total > 0:
        print(f"{dist_cat}: {errors}/{total} = {errors/total*100:.1f}% errors")
