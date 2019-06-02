# ECE471_pca.py
# ECE471 Dr.Qi
# written by Noah Caldwell
# 5/6/19
# Used in final project.

import numpy as np

# # # # # # # # # # # # # # # # #
# PCA Reduction implementation  #
# # # # # # # # # # # # # # # # #

# Return the eigenvalues and eigenvectors of the dataset in descending order of eigenvalue
def pca_eigen(dataset):
    cov = np.cov(dataset.T)
    eigvals, eig = np.linalg.eig(cov)
    return (eigvals, eig)

# Return new basis vectors for a dataset
def pca_find_bases(dataset):
    eigvals, eig = pca_eigen(dataset)

    # Sort eigenvectors by their eigenvalues in descending order
    basis_vectors = [x for _, x in sorted(list(zip(eigvals,eig)))]
    basis_vectors = basis_vectors[::-1]
    return np.array(basis_vectors).astype(np.float)

# Calculate the error associated with the degree of dimensionality reduction
# Returns an array with the proportional error for each value of m
def pca_error(dataset):
    dataset = dataset.T[:-1].T
    # Error can be reduced to the sum of the eigenvalues associated with principal components 
    #   m-1 to d of a d-dimensional data set reduced to m dimensions
    # Therefore the sum of all the eigenvalues makes up 100% of the variability of the dataset
    #   and the eigenvalue of a PC is the proportion of that total which that PC accounts for in the estimation
    totalerror = np.sum(pca_eigen(dataset)[0])
    eigenvalues = sorted(pca_eigen(dataset)[0], reverse=True)
    contributions = [i / totalerror for i in eigenvalues]
    coverage_m = np.cumsum(contributions)

    # An array with the error associated with each m value
    errors = np.subtract(np.full_like(dataset[0], 1.0), coverage_m)
    print("Error for representing data in terms of i dimensions: " + str(errors))
    return errors

# Zoop your x vector onto the new basis vectors, returning y
def pca_zoop(x, bases):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = np.matmul(bases[i].T, x)
    return y

# Represent x in terms of m dimensions on a new set of bases
# Returns the m-dimensional vector in a tuple with the basis vectors
def pca_reduce(x, m, bases):
    if (m > len(x) or m < 1): raise "m must be a positive non-zero integer less or equal to than the dimensionality of x"
    y = pca_zoop(x, bases)
    for i in range (m, len(y)):
        y[i] = 0.0
    return y[0:m]

# Reframes input data set in terms of m dimensions by new basis vectors
def pca(dataset, m):
    classes = dataset.T[-1].T
    dataset = dataset.T[:-1].T
    basis = pca_find_bases(dataset)
    newdata = None

    for i in range(len(dataset)):
        newpoint = pca_reduce(dataset[i], m, basis)
        newpoint = np.append(newpoint, classes[i])

        if (newdata is None):
            newdata = np.array([newpoint])
        else:
            newdata = np.append(newdata, np.array([newpoint]), axis=0)

    return newdata

# Manually remove columns from base dataset
def yeet(dataset, cols):
    cols.sort()
    def isleftmost(col):
        while (col-1) in cols:
            col -= 1
            if col == 0:
                return True
        return False

    removed = 0

    for c in cols:
        c -= removed
        if isleftmost(c):
            dataset = dataset.T[c+1:].T
        else:
            dataset = np.concatenate((dataset.T[0:c].T, dataset.T[c+1:].T), axis=1)
        removed += 1
    
    return dataset

