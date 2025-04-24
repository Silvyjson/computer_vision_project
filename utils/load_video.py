import cv2
import os

def load_video(sequence_path, fps=30):
    """Loads and plays a video sequence from COCO dataset."""
    image_files = sorted(os.listdir(sequence_path))
    delay = int(10000 / fps)
    for img_file in image_files:
        frame = cv2.imread(os.path.join(sequence_path, img_file))
        if frame is None:
            continue

        frame_resized = cv2.resize(frame, (640, 480))
        
        cv2.imshow("Video Frame", frame_resized)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def load_video_frame(sequence_path, target_size=(640, 480)):

    if not os.path.isdir(sequence_path):
        print(f"[ERROR] The path {sequence_path} does not exist.")
        return []

    frames = []
    image_files = sorted(os.listdir(sequence_path))

    for img_file in image_files:
        img_path = os.path.join(sequence_path, img_file)
        frame = cv2.imread(img_path)
        if frame is not None:
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

    print(f"[INFO] Loaded {len(frames)} frames from {sequence_path}")
    return frames
