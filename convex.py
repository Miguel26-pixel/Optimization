import cv2
import numpy as np

# Load the image
image = cv2.imread('cocacola.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw the convex hull
convex_hull_image = np.zeros_like(image)

# Iterate through the contours and find the convex hull
for contour in contours:
    convex_hull = cv2.convexHull(contour)
    cv2.drawContours(convex_hull_image, [convex_hull], 0, (0, 255, 0), 2)

# Display the original image and the convex hull image
cv2.imshow('Original Image', image)
cv2.imshow('Convex Hull Image', convex_hull_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
