import cv2
import time

# Load the cascade for nose detection
nose_cascade = cv2.CascadeClassifier('../resources/xml_files/haarcascade_mcs_nose.xml')

# Read the input image
image = cv2.imread('../resources/photos/profile_photo.jpg')

# Start time
start_time = time.time()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect noses in the image
noses = nose_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangle around the noses
for (x, y, w, h) in noses:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue rectangle

# Display the output
cv2.imshow('Nose Detection', image)

# End time
end_time = time.time()

# Print the time taken
print(f"Time taken: {end_time - start_time} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Windows closed successfully!")