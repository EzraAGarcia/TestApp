import pyrealsense2 as rs
import cv2
import numpy as np

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Define the yellow color range in HSV color space
lower_yellow = np.array([18, 48, 142])
upper_yellow = np.array([36, 93, 255])

# Loop over frames from the camera
while True:
    # Wait for a new frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(np.asarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)

    # Threshold the image to get only yellow pixels
    hsv = cv2.cvtColor(np.asarray(color_frame.get_data()), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine the grayscale and color threshold images using a bitwise AND operation
    combined = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply Gaussian blur to the combined image to reduce noise
    blurred = cv2.GaussianBlur(combined, (5, 5), 0)

    # Apply Canny edge detection algorithm to the blurred image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Apply Hough transform to detect the lines in the edge image
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=100)

    # Draw the detected lines on the color frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(np.asarray(color_frame.get_data()), (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Calculate the angle of the average line
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        avg_angle = np.mean(angles)

        # Display the angle on the screen
        cv2.putText(np.asarray(color_frame.get_data()), "Lane angle: {:.2f}".format(avg_angle),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print(avg_angle)

    # Display the resulting frame
    cv2.imshow('Lane detection', np.asarray(color_frame.get_data()))

    # Wait for a key press and check if it's the ESC key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Clean up
cv2.destroyAllWindows()
pipeline.stop()

