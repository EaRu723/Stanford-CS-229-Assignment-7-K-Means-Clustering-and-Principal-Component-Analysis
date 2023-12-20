import numpy as np

# Z is the projected data in the lower-dimensional space
# U is the matrix of principal components
# K is the number of principal components used in the projection
def recover_data(Z, U, K):
    # Initialize X_rec to store the recovered approximation of the original data
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # Ureduce is the matrix containing the first k columns of U
    # These are the K principal components used for projecting the data
    Ureduce = U[:, np.arange(K)]

    # Recover the original data by projecting back on to the original feature space
    # Multiply the pojected data Z with the transpose of Ureduce
    X_rec = np.dot(Z, Ureduce.T)

    return X_rec
