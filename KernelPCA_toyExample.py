import numpy as np
import matplotlib.pyplot as plot
from kernelPCA_Gaussian import kernelPCA
import math

class kPCA_toy:
    """
    Implementation of toy example in paper "Kernel PCA and De-Noising in feature space" by Mika et. al.
    """
    def __init__(self, mean, var, C, nClusters, nTrainingPoints, nTestPoints, nDim, max_eigVec):
        self.name = "kPCA_toy_example"
        self.mean = mean
        self.var = var
        self.std = np.power(var, 0.5)
        self.C = C
        self.nClusters = nClusters
        self.nTrainingPoints = nTrainingPoints
        self.nTestPoints = nTestPoints
        self.nDim = nDim
        self.max_eigVec = max_eigVec
        self.kPCA = kernelPCA()
        self.training_data = self.kPCA.create_gaussian_data(mean, self.std, nTrainingPoints, nClusters, nDim)\
                                .reshape([nClusters*nTrainingPoints, nDim])
        self.test_data = self.kPCA.create_gaussian_data(mean, self.std, nTestPoints, nClusters, nDim)\
                                .reshape([nClusters*nTestPoints, nDim])

    def gamma_weights(self, V, projection_matrix, nComponents):
        """
        Calculate Gamma for all dataset,
        :param V: normalized vector (alpha_i),
        :param projection_matrix: (T x n) Matrix of beta_k values for all test set,
        :param nComponents: number of components in projection,
        :return: Gamma_i = sum_k( alpha_i * beta_k)
        """
        return np.dot(projection_matrix, (V[:, :nComponents]).transpose())

    def zKernel(self, z_init):
        """
        Calculate Gaussian kernel k(z_init, dataset)
        :param z_init:
        :return:
        """
        z_kernel = np.zeros((self.nClusters*self.nTestPoints, self.nClusters*self.nTrainingPoints), dtype=float)
        for i in range(self.nClusters*self.nTestPoints):
            for j in range(self.nClusters*self.nTrainingPoints):
                z_kernel[i, j] = self.kPCA.Gaussian_Kernel(z_init[i, :], self.training_data[j, :], self.C)

        return z_kernel

    def approximate_input_data(self, gamma, z_init):
        """
        Return updated value of approximated input data z_(t+1)
        :param gamma:
        :param z_init:
        :return:
        """
        z_kernel = self.zKernel(z_init)
        z_num = np.dot(np.multiply(gamma, z_kernel), self.training_data)
        z_den = np.sum(np.multiply(gamma, z_kernel), axis=1)
        return np.divide(z_num, np.repeat(np.matrix(z_den).transpose(), self.nDim, axis=1))

    def kernelPCA_sum(self, nIterations):

        # create Projection matrix for all test points
        kGram, norm_vec = self.kPCA.normalized_eigenVectors(self.training_data, self.C)
        projection_kernel = self.kPCA.projection_kernel(self.training_data, self.test_data, self.C)
        projection_matrix_centered = self.kPCA.projection_centering(kGram, projection_kernel)
        projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :self.max_eigVec])

        # approximate input
        gamma = self.gamma_weights(norm_vec, projection_matrix, self.max_eigVec)
        z_init = np.random.rand(self.nClusters * self.nTestPoints, self.nDim)

        for i in range(nIterations):
            z_init = self.approximate_input_data(gamma, z_init)

        # calculate the mean square distance from cluster center of each point
        z_proj = z_init
        mean_sqr_sum = 0
        for i in range(self.nClusters):
            for j in range(self.nTestPoints):
                mean_sqr_sum += (np.linalg.norm(z_proj[i * self.nTestPoints + j] - mean[i], ord=2) ** 2)

        return mean_sqr_sum

if __name__ == "__main__":
    # create data
    # choose mean uniformly between [-1,1]
    nDimension = 10
    nClusters = 11
    nTrainingPoints = 100
    nTestPoints = 33

    var = 0.1
    max_eigVec = 8
    nIterations = 10
    mean = np.random.uniform(-1, 1, nClusters*nDimension).reshape([nClusters, nDimension])
    var_matrix = np.zeros((nClusters, nDimension, nDimension), dtype=float)
    for j in range(nClusters):
        var_matrix[j, :, :] = (var * np.eye(nDimension, dtype=float))

    # create class object

    kPCAtoy = kPCA_toy(mean, var_matrix, 2 * var, nClusters, nTrainingPoints, nTestPoints, nDimension, max_eigVec)
    mean_sqr_sum = kPCAtoy.kernelPCA_sum(nIterations)

    print("Sum= %f, var=%f, max_eigVec=%d" % (mean_sqr_sum, var, max_eigVec))

    # linear PCA
