import cv2
import os

print("Current working directory:", os.getcwd())

image_path = 'CDLuneScore.png'
print("Full path to the image:", os.path.abspath(image_path))

image = cv2.imread(image_path)

if image is not None:
    print("Image loaded successfully.")
    cv2.imshow('Test Image', image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()
    print("Image window closed.")
else:
    print("Failed to load image. Please check the file path and file integrity.")

print("Exiting program.")
