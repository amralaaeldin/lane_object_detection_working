from utils.object_detection import detect_objects
from utils.lane_detection import detect_lane
import cv2

# Define the input and output video paths
input_video_path = './project_video.mp4'
output_video_path = input_video_path + '_output.mp4'

# Initialize the input and output video streams
input_stream = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = input_stream.get(cv2.CAP_PROP_FPS)
width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_stream = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# Process the video frame by frame
while True:
    # Read the next frame from the input stream
    success, frame = input_stream.read()

    # Stop the loop if we have reached the end of the input stream
    if not success:
        break

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
