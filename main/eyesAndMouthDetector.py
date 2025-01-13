import cv2
import time

# Start time
start_time = time.time()

# Load the cascade for detection on eyes and mouth
eye_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_mcs_mouth.xml')

# Read the input image
image = cv2.imread('../resources/photos/Lenna_(test_image).png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect eyes and mouth in the image
eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
mouths = mouth_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangle around the eyes
for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle

# Draw rectangle around the mouths
for (x, y, w, h) in mouths:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red rectangle

# Display the output
cv2.imshow('Nose Detection', image)

# End time
end_time = time.time()

# Print the time taken
print(f"Time taken: {end_time - start_time} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Windows closed successfully!")