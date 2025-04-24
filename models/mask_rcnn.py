import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import time

class MaskRCNNDetector:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1')
        self.model.eval()

    def detect(self, frames):
        predictions = []
        start = time.time()

        for frame in frames:
            # Convert the frame to tensor format
            image_tensor = F.to_tensor(frame).unsqueeze(0)

            with torch.no_grad():
                # Run the image through the model to get detections
                outputs = self.model(image_tensor)

            preds = []

            for i in range(len(outputs[0]['boxes'])):
                # Extract box coordinates, score, label, and masks
                x1, y1, x2, y2 = outputs[0]['boxes'][i].cpu().numpy()
                score = outputs[0]['scores'][i].cpu().numpy()
                label = outputs[0]['labels'][i].cpu().numpy()

                if score < 0.4:
                    continue

                # Append detection details (bounding box, score, label)
                preds.append({
                    "label": int(label),
                    "confidence": float(score),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

            predictions.append(preds)

        end = time.time()
        print(f"[MaskRCNN] Inference Time: {end - start:.2f} seconds")

        return predictions


class MaskRCNNSegmenter:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1')
        self.model.eval()

    def segment(self, frames):
        predictions = []
        start = time.time()

        for frame in frames:
            # Convert the frame to tensor format
            image_tensor = F.to_tensor(frame).unsqueeze(0)

            with torch.no_grad():
                # Run the image through the model to get segmentation
                outputs = self.model(image_tensor)

            frame_predictions = []

            for i in range(len(outputs[0]['masks'])):
                # Extract the mask, label, and confidence score
                mask = outputs[0]['masks'][i, 0].cpu().numpy()
                label = outputs[0]['labels'][i].cpu().numpy()
                score = outputs[0]['scores'][i].cpu().numpy()

                # Threshold the mask to create a binary mask
                mask_binary = np.where(mask > 0.5, 1, 0).astype(np.uint8)

                if score < 0.4:
                    continue

                # Store the segmented object details
                frame_predictions.append({
                    "label": int(label),
                    "confidence": float(score),
                    "mask": mask_binary
                })

            predictions.append(frame_predictions)

        end = time.time()
        print(f"[MaskRCNN SEGMENTATION] Inference Time: {end - start:.2f} seconds")
        
        return predictions
