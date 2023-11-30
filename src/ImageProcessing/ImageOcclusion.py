import cv2
import numpy as np

image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Retrieve image dimension
height, width, _ = image.shape

# Create occlusion area
# Occlude lower part of the image
occl_start_point = (0, height // 2)
occl_end_point = (width, height)

# Occlusion area color
color = (0, 0, 0)  # Black color
thickness = -1     # Fill the entire rectangle

occluded_image = cv2.rectangle(image, start_point, end_point, color, thickness)

# Save
cv2.imwrite('occluded_image.jpg', occluded_image)

# Display
cv2.imshow('Occluded Image', occluded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
