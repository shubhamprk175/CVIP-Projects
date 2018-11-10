# %% Import Statements
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

UBIT = 'spareek'
np.random.seed(sum([ord(c) for c in UBIT]))
print("Code tested on OpenCV version : 3.4.1")
print("Your OpenCV version: {}".format(cv2.__version__))
OUTPUT_DIR = "./outputs/Task3_output/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %% Method Declarations
def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return True


def save_image(img, fname="test"):
    cv2.imwrite(OUTPUT_DIR+fname+".jpg", img)
    return True


def eu_distance(a, b, ax=1):
    """
    Compute the euclidean distance between a point and the centroid
    :param ax type int: decides the type of norm, 1 means euclidean
    """
    return np.linalg.norm(a - b, axis=ax)


def classify_points(X, centroids):
    """
    Classify each coordinate to specified number of centroids
    :param X type list: List of coordinates
    :param centroids type list: List of Centroids' coordinates
    :return clusters type list: list containing the # of the centroid the point belongs to
    """
    clusters = np.zeros(len(X), np.uint8)
    for i in range(len(X)):
        distances = eu_distance(X[i], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    return clusters


def compute_new_centroid(k, X, clusters):
    """
    Compute new the centroids by taking mean of the previos cluster and assign
    and update the old centroids
    :param k type int: number of centroids
    :param X type list: Flattened image
    :param clusters type list: containing the cluster a point belongs to
    :return centroids type list: updated centroid
    """
    centroids = np.zeros((k, 3))
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


def quantize_image(img1, k):
    img = deepcopy(img1)
    h, w, d = img.shape
    # Flatten the image in 2D - [(w*h), d]
    X = np.reshape(img, (w * h, d))
    # Initialize centroids with first k values of image
    centroids = X[0:k]
    for _ in range(3):
        clusters = classify_points(X, centroids)
        # Finding the new centroids by taking the average value
        centroids = compute_new_centroid(k, X, clusters)

    # Putting the color values of centroid in the image based on
    # the cluster they belong to
    for i in range(len(X)):
        X[i] = centroids[clusters[i]]
    # Getting the image back in 3D shape
    quantized_img = np.reshape(X, (h, w, d))
    show_image(quantized_img)
    save_image(quantized_img, 'task3_baboon_'+str(k))
    print("Done for k=",k)


# %% Image Read and quantize
img = cv2.imread('images/20170414_165344.jpg')
for k in [3, 5, 10, 20]:
    quantize_image(img, k)

# quantize_image(img, 10)
# quantize_image(img, 20)
