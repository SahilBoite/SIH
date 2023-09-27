import cv2
import numpy as np
from ultralytics import YOLO
import time

cap = cv2.VideoCapture(0)

# Set video input resolution to 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set video input frame rate to 24 fps
cap.set(cv2.CAP_PROP_FPS, 24)

model = YOLO("yolov8m.pt")

classes = ['person', 'car', 'motorcycle', 'traffic light', 'stop sign', 'bus', 'truck']

with open("/Users/sahilbhoite/Developer/AIDAS_python/COCO.names.txt", "r") as f:
    class_labels = f.read().splitlines()

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

display_fps = 30  # Desired display frame rate
delay = int(1000 / display_fps)  # Delay between frames in milliseconds
frame_skip = int(round(cap.get(cv2.CAP_PROP_FPS) / display_fps))  # Number of frames to skip
frame_count = 0  # Initialize frame counter
start_time = time.time()  # Start time for FPS calculation

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    results = model(frame, device="mps", conf=0.35)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confidences = np.array(result.boxes.conf.cpu(), dtype="float")

    for cls, bbox, confidence in zip(classes, bboxes, confidences):
        x, y, x2, y2 = bbox
        class_label = class_labels[cls]
        color = colors[cls % len(colors)]  # Assign color based on class index
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        # Change font to Arial for class labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.8
        cv2.putText(frame, f"{class_label} {confidence * 100:.2f}%", (x, y - 5), font, font_size, color, 2, cv2.LINE_AA)

    # Calculate elapsed time for processing each frame
    elapsed_time = time.time() - start_time

    # Calculate actual frame rate
    frame_rate = frame_count / elapsed_time

    # Add frame rate counter to the frame
    cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 30), font, font_size, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(delay) & 0xFF  # Mask the key with 0xFF to get the ASCII value

    if key == ord('Q') or key == ord('q'):  # Break the loop if 'Q' or 'q' is pressed
        break

    # Reset frame count and start time for next calculation
    frame_count = 0
    start_time = time.time()

cap.release()
cv2.destroyAllWindows()
