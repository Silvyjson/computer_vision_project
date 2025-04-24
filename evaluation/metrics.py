import numpy as np
from pycocotools.coco import COCO
import cv2

# ----------------------------
# Detection Metrics
# ----------------------------

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_detection(predictions_per_frame, ground_truth_per_frame, iou_threshold=[0.0,]):
    results_per_threshold = {}
    
    for threshold in iou_threshold:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        iou_scores = []

        for predictions, ground_truth in zip(predictions_per_frame, ground_truth_per_frame):
            tp, fp, fn = 0, 0, 0
            matched_gt = set()

            # print("\n📦 Predictions:")
            # for i, pred in enumerate(predictions[:1]):
            #     print(f"  [{i}] Label: {pred.get('label')}, BBox: {pred.get('bbox')}")

            # # Print all ground truths
            # print("\n🎯 Ground Truths:")
            # for i, gt in enumerate(ground_truth[:1]):
            #     print(f"  [{i}] Label: {gt.get('label')}, BBox: {gt.get('bbox')}")

            for pred in predictions:
                pred_box = pred['bbox']
                pred_label = pred['label']
                match_found = False

                for idx, gt in enumerate(ground_truth):
                    if idx in matched_gt:
                        continue

                    gt_box = gt['bbox']
                    gt_label = gt['label']

                    if pred_label != gt_label:
                        continue

                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= threshold:
                        tp += 1
                        matched_gt.add(idx)
                        match_found = True

                        iou_scores.append(iou)
                        break

                if not match_found:
                    fp += 1

            fn = len(ground_truth) - len(matched_gt)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        results_per_threshold[f"IoU@{threshold:.2f}"] = {
            'IoU': float(np.mean(iou_scores)) if iou_scores else 0.0,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    all_precisions = [results['Precision'] for results in results_per_threshold.values()]
    map_score = float(np.mean(all_precisions))
    
    results_per_threshold['mAP'] = map_score

    return results_per_threshold

def load_ground_truth_detections(coco_annotation_file):
    coco = COCO(coco_annotation_file)
    img_ids = sorted(coco.getImgIds()) 

    all_gt_boxes_per_frame = []

    for img_id in img_ids:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        frame_gt = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            category_id = ann['category_id']
            label = coco.loadCats([category_id])[0]['name']

            frame_gt.append({'label': label, 'bbox': [x1, y1, x2, y2]})

        all_gt_boxes_per_frame.append(frame_gt)

    return all_gt_boxes_per_frame


# ----------------------------
# Segmentation Metrics
# ----------------------------

def calculate_iou_mask(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-6)
    return iou

def dice_coefficient(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    return dice

def evaluate_segmentation(predictions_mask_per_frame, gt_masks_per_frame, iou_threshold=[0.0,]):
    results_per_threshold = {}
    
    for threshold in iou_threshold:
        total_tp, total_fp, total_fn = 0, 0, 0

        iou_scores = []
        dice_scores = []

        for pred_mask_list, gt_mask_list in zip(predictions_mask_per_frame, gt_masks_per_frame):
            tp, fp, fn = 0, 0, 0
            matched_gt = set()

            # print("\n📦 Predictions:")
            # for i, pred in enumerate(pred_mask_list):
            #     print(f"  [{i}] Label: {pred.get('label')}, Mask: {pred.get('mask')}")

            # # Print all ground truths
            # print("\n🎯 Ground Truths:")
            # for i, gt in enumerate(gt_mask_list):
            #     print(f"  [{i}] Label: {gt.get('label')}, Mask: {gt.get('mask')}")

            for pred in pred_mask_list:
                pred_mask = pred['mask']
                pred_label = pred['label']
                match_found = False

                for idx, gt in enumerate(gt_mask_list):
                    if idx in matched_gt:
                        continue

                    gt_mask = gt['mask']
                    gt_label = gt['label']

                    if pred_label != gt_label:
                        continue

                    iou = calculate_iou_mask(pred_mask, gt_mask)

                    if iou >= threshold:
                        matched_gt.add(idx)
                        tp += 1
                        match_found = True

                        iou_scores.append(iou)
                        dice_scores.append(dice_coefficient(pred_mask, gt_mask))
                        break

                if not match_found:
                    fp += 1

            fn = len(gt_mask_list) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        results_per_threshold[f"IoU@{threshold:.2f}"] = {
            'IoU': float(np.mean(iou_scores)) if iou_scores else 0.0,
            'Dice Coefficient': float(np.mean(dice_scores)) if dice_scores else 0.0,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    all_precisions = [results['Precision'] for results in results_per_threshold.values()]
    map_score = float(np.mean(all_precisions))
    
    results_per_threshold['mAP'] = map_score

    return results_per_threshold

def load_ground_truth_segmentation_masks(coco_annotation_file):
    coco = COCO(coco_annotation_file)
    img_ids = sorted(coco.getImgIds()) 

    all_gt_masks = []

    target_size = (480, 640)

    for img_id in img_ids:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

        frame_gt = []

        for ann in annotations:
            if 'segmentation' in ann:                
                mask = coco.annToMask(ann)
                resized_mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

                category_id = ann['category_id']
                label = coco.loadCats([category_id])[0]['name']

                frame_gt.append({'label': label, 'mask': resized_mask.astype(np.uint8)})

        all_gt_masks.append(frame_gt)

    return all_gt_masks