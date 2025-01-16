import cv2
import time
import os

# Function to load cascade classifiers with exception handling
def load_cascade_classifier(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found: {file_path}")
    cascade = cv2.CascadeClassifier(file_path)
    if cascade.empty():
        raise ValueError(f"Failed to load cascade classifier from: {file_path}")
    return cascade

# Load the cascade for face, eyes, mouth, and nose detection
try:
    face_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_frontalface_default.xml')
    eye_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_eye.xml')
    mouth_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_mcs_mouth.xml')
    nose_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_mcs_nose.xml')
except (FileNotFoundError, ValueError) as e:
    print(e)
    exit(1)

# Function to prompt user until a valid image is provided or 'exit' is entered
def get_valid_image():
    while True:
        image_name = input("Please provide the name of an image: ")

        if image_name.lower() == "exit":
            print("Exiting the program...")
            exit(0)

        # Prepend the folder path
        full_path = f"../resources/photos/{image_name}"

        # Attempt to read the image
        image = cv2.imread(full_path)
        if image is not None:
            return image
        else:
            print("Invalid image!")

# Get valid image from the user
image = get_valid_image()

# Start timer after valid input is received
start_time = time.time()

# Convert image to grayscale
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
    # Draw rectangles around the noses
    for (nx, ny, nw, nh) in noses:
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)  # Blue rectangle

# Display the output
cv2.imshow('Face Detection Highlighted', image)

# End timer when the image is displayed
end_time = time.time()

# Print the time taken
print(f"Time taken: {end_time - start_time} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Windows closed successfully!")
