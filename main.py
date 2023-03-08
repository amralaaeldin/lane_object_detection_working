from tensorflow import keras
from scipy.misc import imresize
from ultralytics import YOLO
import numpy as np
import cvzone
import cv2
import time
import math

# Load the lane detection model
lane_model = keras.models.load_model(r'model.h5')

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

# Define the input and output video paths
input_video_path = './project_video.mp4'
output_video_path = './output.mp4'

# Initialize the input and output video streams
input_stream = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = input_stream.get(cv2.CAP_PROP_FPS)
width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_stream = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


lanes = Lanes()

# Process the video frame by frame
while True:
    # Read the next frame from the input stream
    success, frame = input_stream.read()

    # Stop the loop if we have reached the end of the input stream
    if not success:
        break

    # Perform lane detection on the current frame    
    # Preprocess the image (resize, normalize, etc.)
    small_img = imresize(frame, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]
    # Use the lane detection model to predict the lane lines
    prediction = lane_model.predict(small_img)[0] * 255
    lanes.recent_fit.append(prediction)

    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Draw the detected lane lines on the frame
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = imresize(lane_drawn, (720, 1280, 3))
    frame = cv2.addWeighted(frame, 1, lane_image, 1, 0)

    # Use the YOLOv8 model to detect objects in the frame
    detections = yolo_model(frame)
    # Draw the bounding boxes for the detected objects on the frame
    for r in detections:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)
    # Write the output frame to the output stream
    output_stream.write(frame)

    # Display the output frame (optional)
    cv2.imshow('Lane detection and object detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the input and output video streams
input_stream.release()
output_stream.release()

# Close all windows (optional)
cv2.destroyAllWindows()
