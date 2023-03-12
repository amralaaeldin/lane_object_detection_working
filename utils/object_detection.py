from utils.image_utils import *
from ultralytics import YOLO
import math

# Load the YOLOv8 model
yolo_model = YOLO("./yolov8n.pt")

# yolo classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def detect_objects(frame):
  # Use the YOLOv8 model to detect objects in the frame
  detections = yolo_model(frame)
  # Draw the bounding boxes for the detected objects on the frame
  for r in detections:
      boxes = r.boxes
      for box in boxes:
          # Bounding Box
          x1, y1, x2, y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          w, h = x2 - x1, y2 - y1
          # Confidence
          conf = math.ceil((box.conf[0] * 100)) / 100
          # Class Name
          cls = int(box.cls[0])

          put_corner_rect(frame, (x1, y1, w, h))
          text = f'{classNames[cls]} {conf}'
          put_text_rect(frame, text, (max(0, x1), max(35, y1)), scale=1, thickness=2)
  return frame