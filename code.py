import cv2
import numpy as np

# Load the image
image = cv2.imread('cocacola.png', cv2.IMREAD_GRAYSCALE)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Dilation: Expands the white regions in the image
dilated = cv2.dilate(image, kernel, iterations=1)

# Erosion: Shrinks the white regions in the image
eroded = cv2.erode(image, kernel, iterations=1)

# Opening: Erosion followed by dilation (removes noise)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing: Dilation followed by erosion (closes small holes)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Save the results
cv2.imwrite('dilated.jpg', dilated)
cv2.imwrite('eroded.jpg', eroded)
cv2.imwrite('opening.jpg', opening)
cv2.imwrite('closing.jpg', closing)

# Display the images (optional)
cv2.imshow('Dilated', dilated)
cv2.imshow('Eroded', eroded)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()