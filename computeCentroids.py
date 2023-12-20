import numpy as np


def compute_centroids(X, idx, K):
    # returns a tuple representing the dimensions of X (m = number of data points, n = number of features)
    (m, n) = X.shape

    # Initialize an array to store the new centroids, with K rows (one for each centroid) and n columns (one for each feature)
    centroids = np.zeros((K, n))

    for k in range(K):
        # Select all data points assigned to the k-th cluser
        x_for_centroid_k = X[np.where(idx == k)]
        # Compute the mean (centroid) of these points
        centroid_k = np.sum(x_for_centroid_k, axis = 0) / x_for_centroid_k.shape[0]
        # Assign the computed centroid to the k-th row in centroids array
        centroids[k] = centroid_k

    return centroids
