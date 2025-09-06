# Debug Step 1: Check if the click handler is connected

# After running: fig = create_review_plot()
# Add this test:
def debug_click(event):
    print(f"DEBUG: Click detected at {event.xdata}, {event.ydata}")
    if event.inaxes is None:
        print("DEBUG: Click was outside axes")
    else:
        print("DEBUG: Click was inside axes")

# Connect the debug handler
fig.canvas.mpl_connect('button_press_event', debug_click)

###########################################################################

# Debug Step 2: Check if detections are loaded

# Check if current_detections is populated
print(f"Number of detections loaded: {len(current_detections)}")
print("First few detection IDs:", list(current_detections.keys())[:5])

# Check a sample detection's bbox
if current_detections:
    sample_id = list(current_detections.keys())[0]
    sample_det = current_detections[sample_id]
    print(f"Sample detection {sample_id} bbox: {sample_det['bbox']}")

###########################################################################

# Debug Step 3: Test the bbox collision detection

def test_detection_click_manually(click_x, click_y):
    """Manually test if a click point hits any detection"""
    print(f"Testing click at ({click_x}, {click_y})")
    
    for det_id, det_info in current_detections.items():
        bbox = det_info['bbox']
        if bbox[0] <= click_x <= bbox[2] and bbox[1] <= click_y <= bbox[3]:
            print(f"HIT: Detection {det_id} at bbox {bbox}")
            return det_id
    
    print("No detection hit")
    return None

# Test with some coordinates from your image
# Try coordinates near the center of your image
test_detection_click_manually(16000, 16000)  # Adjust these numbers based on your image size
