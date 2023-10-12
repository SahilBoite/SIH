import cv2
import numpy as np
from ultralytics import YOLO
from IPython.display import display, Image
import random

# Replace the following paths with your actual paths
video_path = (0)
classes_path = "/Users/sahilbhoite/Developer/INDRA_SIH/Datasets/classes.txt"
model_path = "/Users/sahilbhoite/Developer/INDRA_SIH/YOLO smoke/best.pt"

cap = cv2.VideoCapture(video_path)

model = YOLO(model_path)

with open(classes_path, "r") as f:
    class_labels = f.read().splitlines()

random.seed(42)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(30)]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, device="mps", conf=0.35)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confidences = np.array(result.boxes.conf.cpu(), dtype="float")

    for cls, bbox, confidence in zip(classes, bboxes, confidences):
        x, y, x2, y2 = bbox
        class_label = class_labels[cls]
        color = colors[cls % len(colors)]
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_label} {confidence * 100:.2f}%", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2,
                    cv2.LINE_AA)

    cv2.imshow("YOLO Output", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
