from scipy.optimize import root_scalar
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import random

def fi(xi, ap, am, z_list, i, j):
    return 2 * (ap**2) * sum(xi - z_i_j for z_i_j in z_list[:j]) + 2 * (am**2) * sum(xi - z_i_j for z_i_j in z_list[j+1:])

def get_intensities_and_weights(G, node):
    intensities = []
    weights = []
    endlist = []
    
    for neighbor in G.neighbors(node):
        edge_data = G[node][neighbor]
        
        # Check if 'intensity' and 'weight' attributes exist in the edge data
        if 'intensity' in G.nodes[neighbor] and 'weight' in edge_data:
            intensity = G.nodes[neighbor]['intensity']
            weight = edge_data['weight']
            
            intensities.append(intensity)
            weights.append(weight)
    
    for i in range(len(intensities)):
        sum = intensities[i]+weights[i]
        endlist.append(sum)

    return endlist


file = input("File: ")

ap = input("alpha_i_plus: ")

am = input("alpha_i_minus: ")

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
            for dx in [-7,-6,-5,-4,-3, -2, -1, 0, 1, 2, 3,4,5,6,7]:
                for dy in [-7,-6,-5,-4,-3, -2, -1, 0, 1, 2, 3,4,5,6,7]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width and neighbor_mask[ni,nj] == 1:
                        maxi = max(abs(dx),abs(dy))
                        G.add_edge((i, j), (ni, nj), weight=maxi)


new_image = gray.copy()
for node in G.nodes:
    z_list = get_intensities_and_weights(G,node)
    z_list = sorted(z_list)
    i,j = node

    random_neg = random.randint(-3000, -1)
    random_pos = random.randint(1, 3000)

    result = root_scalar(fi, args=(random_pos, random_neg, z_list, i, j), bracket=[-255, 255])

    if result.converged:
        print("Zero:", result.root)
        new_image[i][j] = result.root
    else:
        print("No zero found within the specified bracket.")

cv2.imshow('Gray Image', gray)
cv2.imshow('new_image Image', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()