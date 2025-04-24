import cv2
from utils.visualization import draw_bounding_boxes, draw_masks

def process_video(frames, output_path, results, mode="detection"):

    if not frames:
        print(f"[ERROR] No frames provided.")
        return
    
    if len(frames) != len(results):
        print(f"[ERROR] Number of frames and results does not match.")
        print(f"Number of frame {len(frames)}, Number of result {len(results)}")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps= 1
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx, (frame, result) in enumerate(zip(frames, results)):
        if mode == "detection":
            frame = draw_bounding_boxes(frame, result)
        elif mode == "segmentation":
            frame = draw_masks(frame, result)
        out.write(frame)


    out.release()
    cap = cv2.VideoCapture(output_path)
    print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))

    print(f"Processed video saved to {output_path}")

