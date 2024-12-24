import cv2
import os

# Ensure you're in the correct directory
print("Current working directory:", os.getcwd())

# Path to the image
image_path = 'CDLuneScore.png'
print("Full path to the image:", os.path.abspath(image_path))

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded
if image is not None:
    print("Image loaded successfully.")
    cv2.imshow('Test Image', image)

    # Wait for the key press for 1 millisecond inside a loop
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()
    print("Image window closed.")
else:
    print("Failed to load image. Please check the file path and file integrity.")

print("Exiting program.")
