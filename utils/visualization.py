import cv2
import numpy as np

def draw_bounding_boxes(frame, detections):
    """Visualize detections (bounding boxes) on the frame."""
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        score = det.get("score", 0.0)
        label = det.get("label", "Unknown")

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame



def draw_masks(frame, segmentation):
    """Visualize segmentation (masks) on the frame."""
    for seg in segmentation:
        mask = seg['mask']  # Assuming mask is a binary mask (1 for object, 0 for background)
        label = seg['label']  # Class label

        color = (0, 255, 0)  # Green color for mask

        # Create a color mask for visualization (same size as frame)
        color_mask = np.zeros_like(frame)

        # Apply the mask color (where mask is 1)
        color_mask[mask == 1] = color
        
        # Overlay the color mask on the frame
        frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)
        
        # Add label text for segmentation (e.g., Class ID or Name)
        label_text = f"Class {label}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame
