import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import math


file = input("File: ")

operation = input("Which operation: ")

# Load the image
image = cv2.imread(file)
cv2.imshow('Original Image', image)
cv2.waitKey(0)


height, width, channels = image.shape

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, threshold1=100, threshold2=200)

# Threshold the image to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Define neighborhood criteria (8-connectivity)
neighborhood_criteria = 8

# Calculate pixel intensity differences
differences = cv2.filter2D(gray, -1, np.ones((3, 3))) - 9 * gray

# Define a threshold (adjust as needed)
threshold = 10

# Create a binary mask of neighbors
neighbor_mask = (differences < threshold).astype(int)

# Create a directed graph using NetworkX
G = nx.DiGraph()

height, width = gray.shape
for i in range(height):
    for j in range(width):
        if neighbor_mask[i, j] == 1:
            G.add_node((i, j), intensity=gray[i, j])
            for dx in [-5,-4,-3, -2, -1, 0, 1, 2, 3,4,5]:
                for dy in [-5,-4,-3, -2, -1, 0, 1, 2, 3,4,5]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width and neighbor_mask[ni,nj] == 1:
                        maxi = max(abs(dx),abs(dy))
                        G.add_edge((i, j), (ni, nj), weight=maxi)

new_image = gray.copy()
for node in G.nodes:
    x,y = node
    intensity = G.nodes[node]['intensity']
    # Find the neighbors connected by edges
    neighbor_nodes = list(G.neighbors(node))

    # Find the neighbor with the minimum intensity
    m_intensity = intensity

    for neighbor in neighbor_nodes:
        i,j = neighbor
        neighbor_intensity = G.nodes[neighbor]['intensity']
        if G.has_edge((x,y), (i,j)):
            edge_data = G[(x,y)][(i,j)]
            edge_value = edge_data['weight']
        else: 
            continue
        if (operation == 1):
            if neighbor_intensity - edge_value < m_intensity:
                m_intensity = neighbor_intensity - edge_value
        else:
            if neighbor_intensity + edge_value > m_intensity:
                m_intensity = neighbor_intensity - edge_value

    # Update the intensity of the current pixel with the minimum found
    G.nodes[node]['intensity'] = m_intensity
    print(new_image[x,y], m_intensity)
    new_image[x,y] = m_intensity


print(G)
cv2.imshow('Gray Image', gray)
cv2.imshow('new_image Image', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
