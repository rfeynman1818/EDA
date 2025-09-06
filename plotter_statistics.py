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
