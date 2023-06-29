from utils.object_detection import detect_objects
from utils.lane_detection import detect_lane
import cv2
import datetime

# Initialize the input and output video streams
input_video_path = './laned_eg_timelapse_compressed.m4v'
input_stream = cv2.VideoCapture(input_video_path)

# to read from webcam
""" 
input_stream = cv2.VideoCapture(0)
address = "http://192.168.1.15:8080/video"
input_stream.open(address) 
"""

video_format = input_stream.get(cv2.CAP_PROP_FORMAT)

format_extensions = {
    0: ".avi", 1: ".mjpeg", 2: ".mp4", 3: ".mov", 4: ".mpeg", 5: ".wmv",
    6: ".mkv", 7: ".flv", 8: ".webm", 9: ".h264", 10: ".ogv",
    11: ".vob", 12: ".ts", 13: ".3gp", 14: ".m4v", 15: ".divx",
    16: ".asf", 17: ".m2ts", 18: ".mpg", 19: ".swf", 20: ".f4v",
}

output_video_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_output' + format_extensions.get(video_format, "")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = input_stream.get(cv2.CAP_PROP_FPS)
# width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 1280
height = 720
output_stream = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process the video frame by frame
while True:
    # Read the next frame from the input stream
    success, frame = input_stream.read()

    # Stop the loop if we have reached the end of the input stream
    if not success:
        break

    frame = cv2.resize(frame, (width, height))
    frame = detect_lane(frame)

    frame = detect_objects(frame)

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
