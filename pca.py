import numpy as np
import scipy


def pca(X):
    # m: number of examples, n: number of features
    (m, n) = X.shape

    # Initialize U and S which will store the principal components and singular values
    U = np.zeros(n)
    S = np.zeros(n)

    # Compute the covariance matrix of the features
    sigma = np.dot(X.T, X) / m

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    # U will contain principal components (eigenvectors of the covariance matrix)
    # S will contain the singular values (square roots of the eigenvalues of the covariance matrix)
    U, S, _ = scipy.linalg.svd(sigma, full_matrices = True, compute_uv = True)

    return U, S
