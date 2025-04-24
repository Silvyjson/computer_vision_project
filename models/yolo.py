from ultralytics import YOLO
import cv2
import numpy as np
import time

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frames):
        predictions = []
        start = time.time()

        for frame in frames:
            result = self.model.predict(frame, conf=0.4, augment=True, verbose=False)[0]
            preds = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                preds.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
            predictions.append(preds)

        end = time.time()
        print(f"[YOLOv8] Inference Time: {end - start:.2f} seconds")
        return predictions


class YOLOv8Segmenter:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def segment(self, frames):
        predictions = []
        start = time.time()

        for frame in frames:
            result = self.model(frame, conf=0.4, verbose=False)[0]
            segmentation = []

            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    class_id = int(result.boxes.cls[i])
                    label = self.class_names[class_id]
                    conf = float(result.boxes.conf[i])
                    
                    mask_np = mask.cpu().numpy()
                    resized_mask = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                    binary_mask = (resized_mask > 0.5).astype(np.uint8)

                    segmentation.append({
                        "label": label,
                        "confidence": conf,
                        "mask": binary_mask
                    })

            predictions.append(segmentation)

        end = time.time()
        print(f"[YOLOv8 SEGMENTATION] Inference Time: {end - start:.2f}s")
        return predictions
