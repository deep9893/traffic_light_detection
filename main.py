import cv2
import numpy as np


# Load pre-trained vehicle detection model (e.g., Haar cascades or deep learning models)
car_cascade = cv2.CascadeClassifier("D:\\Codebook\\exam\\harcascade file\\haarcascade_car.xml")

# Load pre-trained signal detection model or implement signal detection algorithm
signal_cascade = cv2.CascadeClassifier("D:\\Codebook\\Traffic_light_detection\\cascade.xml")

# Define color ranges for red, yellow, and green signals (adjust as per your requirements)
red_lower = np.array([0, 0, 100], dtype=np.uint8)
red_upper = np.array([20, 255, 255], dtype=np.uint8)
yellow_lower = np.array([20, 0, 100], dtype=np.uint8)
yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
green_lower = np.array([40, 0, 100], dtype=np.uint8)
green_upper = np.array([70, 255, 255], dtype=np.uint8)

# Define ROI coordinates (adjust as per your requirements)
roi_x, roi_y, roi_width, roi_height = 100, 100, 200, 200

# Initialize video capture from a camera or video file
video_path = "D:\\Codebook\\Traffic_light_detection\\demo.avi"
cap = cv2.VideoCapture(video_path)

while True:
    # Read the current frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Convert ROI to the HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Perform color thresholding to detect red, yellow, and green signals
    mask_red = cv2.inRange(hsv_roi, red_lower, red_upper)
    mask_yellow = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv_roi, green_lower, green_upper)

    # Count the number of non-zero pixels in each mask
    red_pixels = np.count_nonzero(mask_red)
    yellow_pixels = np.count_nonzero(mask_yellow)
    green_pixels = np.count_nonzero(mask_green)

    # Determine the signal state based on the number of pixels in each mask
    signal_state = None
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        signal_state = "Red"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        signal_state = "Yellow"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        signal_state = "Green"

    # Draw the signal state text on the frame
    cv2.putText(frame, f"Signal State: {signal_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with the ROI and signal state
    cv2.imshow('Signal State Analysis', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
