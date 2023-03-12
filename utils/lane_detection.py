import numpy as np
from scipy.misc import imresize
from tensorflow import keras
import cv2

# Load the lane detection model
lane_model = keras.models.load_model(r'model.h5')

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


lanes = Lanes()

def detect_lane(frame):
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
  
  return cv2.addWeighted(frame, 1, lane_image, 1, 0)