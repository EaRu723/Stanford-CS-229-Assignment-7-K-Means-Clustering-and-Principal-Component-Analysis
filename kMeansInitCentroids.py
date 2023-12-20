import numpy as np


def kmeans_init_centroids(X, K):
    # Initialize array to store the centroids, with K rows (centroids) and as many columns as features in X
    centroids = np.zeros((K, X.shape[1]))

    # Randomly select K indices from the range of the number of datapoints in X
    indices = np.random.randint(X.shape[0], size = K)
    # Assign the data points at these indices as the inital centroids
    centroids = X[indices]

    return centroids
