#%% Import Statments
%matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
UBIT = 'spareek'
np.random.seed(sum([ord(c) for c in UBIT]))
OUTPUT_DIR = "./outputs/Task3_output/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#%% Methods Declaration
def dist(a, b, ax=1):
    """
    Compute the euclidean distance between a point and the centroid
    :param ax type int: axis
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
        distances = dist(X[i], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    return clusters

def compute_new_centroid(k, X, clusters):
    """
    Compute new the centroids by taking mean of the previos cluster and assign
    and update the old centroids
    :param k type int: number of centroids
    :param X type list: Points list
    :param clusters type list: containing the cluster a point belongs to
    :return centroids type list: updated centroid
    """
    centroids = np.zeros((k, 2))
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


def plot_points(X, clusters, ax, colors=None):
    for i, cluster in enumerate(clusters):
        ax.scatter(X[i][0], X[i][1], marker='^', s=200, edgecolors='b', facecolors=colors[cluster])


def plot_centroids(centroids, ax, colors=['r', 'b', 'g']):
    for i, c in enumerate(colors):
        ax.scatter(centroids[i][0], centroids[i][1], marker='o', s=200, c=c)



#%% Data Reading and inditialization
data = pd.read_csv('data.csv')
f1 = data['x'].values
f2 = data['y'].values
X = np.array(list(zip(f1, f2)))


# Number of clusters
k = 3
# Initialize colors of centroids
colors = ['r', 'b', 'g']
# X coordinates of centroids
C_x = [6.2, 6.6, 6.5]
# Y coordinates of centroids
C_y = [3.2, 3.7, 3.0]
# List of (x, y) coordinates of centroids
centroids = np.array(list(zip(C_x, C_y)), dtype=np.float32)
# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)


#%% First Task
# Cluster Lables(0, 1, 2)
clusters = classify_points(X, centroids)

# Plot centroids based on their colors
fig, ax = plt.subplots()
plot_centroids(centroids, ax, colors=colors)
plot_points(X, clusters, ax, colors=colors)
fig.savefig(OUTPUT_DIR + 'task3_iter1_a.jpg')
print(clusters)


#%% Second Task

# Storing the old centroid values
centroids_old = deepcopy(centroids)
# Finding the new centroids by taking the average value
centroids = compute_new_centroid(k, X, clusters)

fig, ax = plt.subplots()
plot_centroids(centroids, ax, colors=colors)
fig.savefig(OUTPUT_DIR + 'task3_iter1_b.jpg')

print(centroids)
#%% Third Task

clusters = classify_points(X, centroids)

fig, ax = plt.subplots()
plot_points(X, clusters, ax=ax, colors=colors)
fig.savefig(OUTPUT_DIR + 'task3_iter2_a.jpg')


# Storing the old centroid values
centroids_old = deepcopy(centroids)
# Finding the new centroids by taking the average value
centroids = compute_new_centroid(k, X, clusters)

fig, ax = plt.subplots()
# Plot newly computed centroids based on their colors
plot_centroids(centroids, ax, colors=colors)
fig.savefig(OUTPUT_DIR + 'task3_iter2_b.jpg')
print(clusters)
print(centroids)


#
# # Plot centroids based on their colors
# fig, ax = plt.subplots()
# plot_centroids(centroids, ax=ax, colors=colors)
# plot_points(X, clusters, ax=ax, colors=colors)
# fig.savefig('./Task3_output/task3_iter2_b.jpg')
