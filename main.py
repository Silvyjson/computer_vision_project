from utils.load_video import load_video, load_video_frame
from models.yolo import YOLOv8Detector, YOLOv8Segmenter
from models.fast_rcnn import FastRCNNDetector
from models.mask_rcnn import MaskRCNNDetector, MaskRCNNSegmenter
from evaluation.metrics import load_ground_truth_detections, load_ground_truth_segmentation_masks, evaluate_detection, evaluate_segmentation
from utils.convert_label import convert_label_ids_to_names
from utils.process_video import process_video


# Load and play a sample video sequence
sequence_path = "datasets/fiftyone/coco-2017/validation/data/"
# load_video(sequence_path)

def main():
    # Instantiate models
    yolo_detector = YOLOv8Detector()
    yolo_segmenter = YOLOv8Segmenter()
    fast_rcnn_detector = FastRCNNDetector()
    mask_rcnn_detector = MaskRCNNDetector()
    mask_rcnn_segmenter = MaskRCNNSegmenter()

    frames = load_video_frame(sequence_path)
    if not frames:
        print(f"[ERROR] No frames loaded from {sequence_path}. Exiting.")
        return
    
    print(f"Processing: {len(frames)} frames from  {sequence_path}")
       
    # Run detection on frames for each model
    print("Running detection with YOLOv8...")
    yolo_predictions = yolo_detector.detect(frames)
    print("Running detection with Fast R-CNN...")
    fast_rcnn_predictions = fast_rcnn_detector.detect(frames)
    print("Running detection with Mask R-CNN...")
    mask_rcnn_predictions = mask_rcnn_detector.detect(frames)
    
    # Run segmentation where applicable
    print("Running segmentation with YOLOv8...")
    yolo_segmentations = yolo_segmenter.segment(frames)
    print("Running segmentation with Mask R-CNN...")
    mask_rcnn_segmentations = mask_rcnn_segmenter.segment(frames)


    coco_annotation_file = 'datasets/fiftyone/coco-2017/validation/labels.json'

    # Load ground truth detection and segmentation masks
    ground_truth_detections = load_ground_truth_detections(coco_annotation_file)
    ground_truth_segmentation_masks = load_ground_truth_segmentation_masks(coco_annotation_file)

    fast_rcnn_predictions = convert_label_ids_to_names(coco_annotation_file, fast_rcnn_predictions)
    mask_rcnn_predictions = convert_label_ids_to_names(coco_annotation_file, mask_rcnn_predictions)
    mask_rcnn_segmentations = convert_label_ids_to_names(coco_annotation_file, mask_rcnn_segmentations)



    # Evaluate detection
    iou_thresh = [0.0, 0.3, 0.5, 0.75]

    detection_metrics_yolo = evaluate_detection(yolo_predictions, ground_truth_detections, iou_threshold=iou_thresh)
    detection_metrics_fast_rcnn = evaluate_detection(fast_rcnn_predictions, ground_truth_detections, iou_threshold=iou_thresh)
    detection_metrics_mask_rcnn = evaluate_detection(mask_rcnn_predictions, ground_truth_detections, iou_threshold=iou_thresh)
    
    segmentation_metrics_yolo = evaluate_segmentation(yolo_segmentations, ground_truth_segmentation_masks, iou_threshold=iou_thresh)
    segmentation_metrics_mask_rcnn = evaluate_segmentation(mask_rcnn_segmentations, ground_truth_segmentation_masks, iou_threshold=iou_thresh)

    print(f"\n📏 Evaluation at IoU Threshold = {iou_thresh}")
    print("Detection Metrics for Yolo:", detection_metrics_yolo)
    print("Detection Metrics for Fast R-CNN:", detection_metrics_fast_rcnn)
    print("Detection Metrics for Mask R-CNN:", detection_metrics_mask_rcnn)

    print("Segmentation Metrics for Yolo:", segmentation_metrics_yolo)
    print("Segmentation Metrics for Mask R-CNN:", segmentation_metrics_mask_rcnn)


    # process_video(frames, "output/yolo_detection.mp4", yolo_predictions, "detection")
    # process_video(frames, "output/yolo_segmentation.mp4", yolo_segmentations, "segmentation")


if __name__ == "__main__":
    main()