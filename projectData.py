import numpy as np

# X is the original dta matrix (examples x features)
# U is the matrix of principal components
# K is the number of principal components to retain
def project_data(X, U, K):
    # Initialize Z - the pojection of X onto the first K principal components
    Z = np.zeros((X.shape[0], K))

    # matrix containing the first K columns of U, reducing the feature space from N-dimensions to K-dimensions
    Ureduce = U[:, np.arange(K)]

    # Project the data into reduced feature space
    Z = np.dot(X, Ureduce)

    return Z
