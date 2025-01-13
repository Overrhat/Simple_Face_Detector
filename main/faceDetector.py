import cv2
import time

# Start time
start_time = time.time()

# Load the cascade for face, eyes, and mouth detection
face_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_mcs_nose.xml')

# Read the input image
image = cv2.imread('../resources/photos/Lenna_(test_image).png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Loop over each detected face
for (fx, fy, fw, fh) in faces:
    # Define the region of interest (ROI) for eyes and mouth within the detected face
    roi_gray = gray[fy:fy+fh, fx:fx+fw]
    roi_color = image[fy:fy+fh, fx:fx+fw]

    # Detect eyes within the ROI
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    # Draw rectangles around the eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green rectangle

    # Detect mouth within the ROI
    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)
    # Draw rectangles around the mouths
    for (mx, my, mw, mh) in mouths:
        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)  # Red rectangle

    # Detect nose within the ROI
    noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
    # Draw rectangles around the mouths
    for (nx, ny, nw, nh) in noses:
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)  # Blue rectangle

# Display the output
cv2.imshow('Face Detection Highlighted', image)

# End time
end_time = time.time()

# Print the time taken
print(f"Time taken: {end_time - start_time} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Windows closed successfully!")
