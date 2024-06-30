# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate some random data
np.random.seed(0)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60)

# Plotting the data points
plt.scatter(X[:, 0], X[:, 1])
plt.title('Generated Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Perform K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Getting the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters and their centers
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, color='red', label='Cluster Centers')
plt.title('Clustered Data with Centers')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
