import numpy as np


def find_closest_centroids(X, centroids):
    # K represents the number of centroids
    K = centroids.shape[0]

    # m is the number of datapoints in X
    m = X.shape[0]

    # idx stores the index of the closest centroid for each data point
    idx = np.zeros(m)

    # temporary array to store distances of each point from each centroid
    means = np.zeros((m,K))

    # Iterate over X and compute the difference between this point and each centroid
    for i in range(m):
        x = X[i]
        diff = x - centroids

        # Compute the distance from the k-th centroid
        for k in range(K):
            means[i,k] = np.linalg.norm(diff[k])

    # For each data point finds the index of the closest centroid
    # The argmin function returns the indices of the minimum values along an axis
    idx = np.argmin(means, axis = 1)

    return idx
