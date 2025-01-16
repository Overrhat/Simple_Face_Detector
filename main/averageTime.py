import cv2
import time
import os
import numpy as np

def load_cascade_classifier(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found: {file_path}")
    cascade = cv2.CascadeClassifier(file_path)
    if cascade.empty():
        raise ValueError(f"Failed to load cascade classifier from: {file_path}")
    return cascade

def run_face_detection():
    # Load the cascade for face, eyes, mouth, and nose detection
    try:
        face_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_frontalface_default.xml')
        eye_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_eye.xml')
        mouth_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_mcs_mouth.xml')
        nose_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_mcs_nose.xml')
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return None

    # Read the input image
    try:
        image = cv2.imread('../resources/photos/lenna.png')
        if image is None:
            raise ValueError("Image not found or unable to read.")
    except ValueError as e:
        print(e)
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over each detected face
    for (fx, fy, fw, fh) in faces:
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        roi_color = image[fy:fy+fh, fx:fx+fw]

        # Detect eyes within the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect mouth within the ROI
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        # Detect nose within the ROI
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

    return image

# Run the face detection 100 times and measure the time
times = []
total_start_time = time.time()  # Start time for all runs

for i in range(100):
    start_time = time.time()
    result = run_face_detection()
    end_time = time.time()
    if result is not None:
        times.append(end_time - start_time)
    print(f"Run {i+1}: {end_time - start_time:.4f} seconds")

total_end_time = time.time()  # End time for all runs
total_execution_time = total_end_time - total_start_time

# Calculate and print the average time and total execution time
if times:
    average_time = np.mean(times)
    print(f"\nAverage time taken: {average_time:.4f} seconds")
    print(f"Total execution time for all runs: {total_execution_time:.4f} seconds")
else:
    print("No successful runs to calculate average time.")

# Display the last processed image
if result is not None:
    cv2.imshow('Face Detection Highlighted', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Windows closed successfully!")
