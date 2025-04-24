import torch
import torchvision
from torchvision.transforms import functional as F
import time

class FastRCNNDetector:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
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
                # Extract box coordinates, score, and label
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
        print(f"[FastRCNN] Inference Time: {end - start:.2f} seconds")
        
        return predictions
