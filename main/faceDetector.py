import cv2
import time
import os
import concurrent.futures

def load_cascade_classifier(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found: {file_path}")
    cascade = cv2.CascadeClassifier(file_path)
    if cascade.empty():
        raise ValueError(f"Failed to load cascade classifier from: {file_path}")
    return cascade

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def detect_feature(cascade, roi, scale_factor, min_neighbors, min_size=(30, 30)):
    return cascade.detectMultiScale(roi, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

def get_valid_image():
    while True:
        image_name = input("Please provide the name of an image in resources (type exit to quit): ")
        if image_name.lower() == "exit":
            print("Closing the program...")
            exit(0)
        full_path = f"../resources/photos/{image_name}"
        image = cv2.imread(full_path)
        if image is not None:
            return image
        else:
            print("Invalid image! Try again!")

def resize_for_display(image, max_width=1024, max_height=768):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def filter_detections(detections, max_detections):
    if len(detections) > max_detections:
        return sorted(detections, key=lambda d: d[2] * d[3], reverse=True)[:max_detections]
    return detections

# Load cascades
try:
    face_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_frontalface_default.xml')
    eye_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_eye.xml')
    mouth_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_mcs_mouth.xml')
    nose_cascade = load_cascade_classifier('../resources/xml_files/haarcascade_mcs_nose.xml')
except (FileNotFoundError, ValueError) as e:
    print(e)
    exit(1)

# Get valid image and start timer
image = get_valid_image()
start_time = time.time()

# Preprocess image
preprocessed = preprocess_image(image)

# Detect faces with more relaxed parameters
faces = detect_feature(face_cascade, preprocessed, scale_factor=1.05, min_neighbors=6)

# Process each face
for (fx, fy, fw, fh) in faces:
    face_roi = preprocessed[fy:fy+fh, fx:fx+fw]
    roi_color = image[fy:fy+fh, fx:fx+fw]

    # Use multi-threading for feature detection
    with concurrent.futures.ThreadPoolExecutor() as executor:
        eye_future = executor.submit(detect_feature, eye_cascade, face_roi[int(fh*0.1):int(fh*0.6), :], 1.05, 6)
        mouth_future = executor.submit(detect_feature, mouth_cascade, face_roi[int(fh*0.5):, :], 1.1, 20)
        nose_future = executor.submit(detect_feature, nose_cascade, face_roi[int(fh*0.2):int(fh*0.8), :], 1.05, 4)

        eyes = eye_future.result()
        mouths = mouth_future.result()
        noses = nose_future.result()

    # Filter detections
    eyes = filter_detections(eyes, 2)
    mouths = filter_detections(mouths, 1)
    noses = filter_detections(noses, 1)

    # Draw rectangles
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey+int(fh*0.1)), (ex + ew, ey + eh+int(fh*0.1)), (0, 255, 0), 2)

    for (mx, my, mw, mh) in mouths:
        cv2.rectangle(roi_color, (mx, my+int(fh*0.5)), (mx + mw, my + mh+int(fh*0.5)), (0, 0, 255), 2)

    for (nx, ny, nw, nh) in noses:
        cv2.rectangle(roi_color, (nx, ny+int(fh*0.2)), (nx + nw, ny + nh+int(fh*0.2)), (255, 0, 0), 2)

# Resize and display
display_image = resize_for_display(image)
cv2.imshow('Face Detection Highlighted', display_image)
cv2.setWindowProperty('Face Detection Highlighted', cv2.WND_PROP_TOPMOST, 1)

# End timer and print time taken
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Windows closed successfully!")