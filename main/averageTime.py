import cv2
import time
import os
import concurrent.futures

def load_classifier(xml_file_path):
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    classifier = cv2.CascadeClassifier(xml_file_path)
    if classifier.empty():
        raise ValueError(f"Failed to load classifier from: {xml_file_path}")
    return classifier

def convert_to_grayscale_and_equalize(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(grayscale_image)
    return equalized_image

def detect_objects(cascade_classifier, region_of_interest, scale_factor, min_neighbors, min_size=(30, 30)):
    return cascade_classifier.detectMultiScale(region_of_interest, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

def prompt_for_valid_image():
    while True:
        image_filename = input("Please provide the name of an image in resources (type exit to quit): ")
        if image_filename.lower() == "exit":
            print("Closing the program...")
            exit(0)
        image_path = f"../resources/photos/{image_filename}"
        loaded_image = cv2.imread(image_path)
        if loaded_image is not None:
            return loaded_image
        else:
            print("Invalid image! Try again!")

def scale_image_for_display(image, max_width=1024, max_height=768):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def limit_detections(detected_objects, max_allowed):
    if len(detected_objects) > max_allowed:
        return sorted(detected_objects, key=lambda obj: obj[2] * obj[3], reverse=True)[:max_allowed]
    return detected_objects

def process_image(input_image):
    start_time = time.time()

    # Preprocess image
    enhanced_image = convert_to_grayscale_and_equalize(input_image)

    # Detect faces
    faces_detected = detect_objects(face_classifier, enhanced_image, scale_factor=1.1, min_neighbors=7)

    # Process each detected face
    for (face_x, face_y, face_width, face_height) in faces_detected:
        face_region_grayscale = enhanced_image[face_y:face_y + face_height, face_x:face_x + face_width]
        face_region_color = input_image[face_y:face_y + face_height, face_x:face_x + face_width]

        # Use multi-threading for feature detection
        with concurrent.futures.ThreadPoolExecutor() as executor:
            eyes_future = executor.submit(detect_objects, eye_classifier, face_region_grayscale[int(face_height*0.1):int(face_height*0.6), :], 1.05, 6)
            mouth_future = executor.submit(detect_objects, mouth_classifier, face_region_grayscale[int(face_height*0.5):, :], 1.07, 20)
            nose_future = executor.submit(detect_objects, nose_classifier, face_region_grayscale[int(face_height*0.2):int(face_height*0.8), :], 1.2, 7)

            detected_eyes = eyes_future.result()
            detected_mouths = mouth_future.result()
            detected_noses = nose_future.result()

        # Limit number of detections
        detected_eyes = limit_detections(detected_eyes, 2)
        detected_mouths = limit_detections(detected_mouths, 1)
        detected_noses = limit_detections(detected_noses, 1)

        # Draw detection rectangles
        for (eye_x, eye_y, eye_width, eye_height) in detected_eyes:
            cv2.rectangle(face_region_color, (eye_x, eye_y + int(face_height*0.1)), (eye_x + eye_width, eye_y + eye_height + int(face_height*0.1)), (0, 255, 0), 2)

        for (mouth_x, mouth_y, mouth_width, mouth_height) in detected_mouths:
            cv2.rectangle(face_region_color, (mouth_x, mouth_y + int(face_height*0.5)), (mouth_x + mouth_width, mouth_y + mouth_height + int(face_height*0.5)), (0, 0, 255), 2)

        for (nose_x, nose_y, nose_width, nose_height) in detected_noses:
            cv2.rectangle(face_region_color, (nose_x, nose_y + int(face_height*0.2)), (nose_x + nose_width, nose_y + nose_height + int(face_height*0.2)), (255, 0, 0), 2)

    # Scale and display image
    output_image = scale_image_for_display(input_image)
    cv2.imshow('Detected Facial Features', output_image)
    cv2.setWindowProperty('Detected Facial Features', cv2.WND_PROP_TOPMOST, 1)

    end_time = time.time()
    return end_time - start_time

# Load classifiers from XML files
try:
    face_classifier = load_classifier('../resources/xml_files/haarcascade_frontalface_default.xml')
    eye_classifier = load_classifier('../resources/xml_files/haarcascade_eye.xml')
    mouth_classifier = load_classifier('../resources/xml_files/haarcascade_mcs_mouth.xml')
    nose_classifier = load_classifier('../resources/xml_files/haarcascade_mcs_nose.xml')
except (FileNotFoundError, ValueError) as error:
    print(error)
    exit(1)

# Load image
input_image = prompt_for_valid_image()

# Run the process 10 times and calculate average
total_time = 0
num_iterations = 10

for i in range(num_iterations):
    processing_time = process_image(input_image.copy())
    total_time += processing_time
    print(f"Iteration {i+1} Processing Time: {processing_time:.2f} seconds")

average_time = total_time / num_iterations
print(f"\nAverage Processing Time: {average_time:.2f} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Display windows closed successfully!")