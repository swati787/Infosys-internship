import cv2
import os

video_path = "../data/raw_videos/cycling.mp4"
output_dir = "../data/frames/cycling"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_name = f"frame_{frame_count}.jpg"
    frame_path = os.path.join(output_dir, frame_name)

    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()
print(f"✅ {frame_count} frames extracted successfully")
