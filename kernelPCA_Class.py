import numpy as np
import matplotlib.pyplot as plot
import math
import multiprocessing
import matplotlib.cm as cm
import pdb


class kernelPCA:
    """
    KernelPCA class to define basic functionality.
    I will present a simple example of Kernel PCA using Gaussians here. The example with try to reproduce the results
    shown in Schölkopf [1998] paper (fig4). The data is generated in 2D using 3 Gaussians in the region [-1,1]x[-0.5,1].
    The standard deviation is 0.1. And we will use Gaussian kernel with C=0.1.
    Method:
    1. Generate data in 2D as specified
    2. Create kernel matrix K
    3. Compute centered kernel matrix Kc
    4. Use PCA to computer eigenvalues and eigenvectors for Kc
    5. Normalize eigenvectors
    6. Use first 8 eigenvectors with maximum eigenvalues and display contours along which the projection onto the
       corresponding principal component is constant.
    """

    def __init__(self):
        self.name = 'kernelPCA'

    def create_gaussian_data(self, mean, std, nPoints, nClusters, nDimension):
        """
        Create Gaussian data with specified parameters
        :param mean: mean of the gaussians for each cluster
        :param std: standard deviation of all gaussians
        :param nPoints: number of points of data in each cluster
        :param nClusters: number of clusters
        :param nDimension: dimension of data in each cluster
        :return: complete dataset
        """
        dataset = np.zeros((nClusters, nPoints, nDimension), dtype=np.float64)
        for i in range(nClusters):
            cov = std[i] ** 2
            dataset[i, :, :] = np.random.multivariate_normal(mean[i], cov, nPoints)

        return dataset

    def plot_data(self, dataset, plt):
        """
        plot dataset onto a figure plt
        :param dataset:
        :return: plot object
        """
        cluster_markers = ['*', '+', 'o']
        cluster_color = ['b', 'g', 'r']
        for i in range(dataset.shape[0]):
            plt.scatter(*zip(*dataset[i]), marker=cluster_markers[i], c=cluster_color[i])

        return plt

    def Kernel_Gram(self, dataset, C):
        """
        Return gaussian or linear kernel gram matrix for given dataset and sigma value
        k(x,x') = exp(-||x - x'||² / C) or k(x,x')= xy -C
        :param dataset: data
        :param C: sigma / constant
        :return: gram matrix K
        """
        N = dataset.shape[0]
        K = np.zeros((N, N), dtype=np.float64)
        XX, YY = np.meshgrid(list(range(N)), list(range(N)))
        XX = XX.reshape([N * N, 1])
        YY = YY.reshape([N * N, 1])
        ZZ = np.column_stack([XX, YY])
        for i in range(ZZ.shape[0]):
            vec = ZZ[i]
            K[vec[0], vec[1]] = self.kernel(dataset[vec[0]], dataset[vec[1]], C)

        return K

    def Kernel_Centering(self, K):
        """
        Centering the K matrix
        :param K:
        :return: K_centered
        """
        N = K.shape[0]
        one_N = np.ones((N, N), dtype=np.float64) / N
        K_centered = K - np.dot(one_N, K) - np.dot(K, one_N) + np.dot(one_N, np.dot(K, one_N))
        return K_centered

    def normalize_eigVec(self, V, L):
        """
        Normalize eigenvectors of K matrix
        :param V: eigen vectors
        :param L: eigen values
        :return: normalized vectors
        """
        N, M = V.shape
        V_norm = np.zeros((V.shape), dtype=np.float64)
        for i in range(M):
            V_norm[:, i] = V[:, i] / math.sqrt(abs(L[i]))

        return V_norm

    def projection_kernel(self, dataset, testset, C):
        """
        Return projection of a point x on vector Vi
        :param dataset: dataset to calculate kernel
        :param testset: test data vector to project
        :param C: sigma value
        :return: Projection value
        """
        N = dataset.shape[0]
        D = testset.shape[0]
        K = np.zeros((D, N), dtype=np.float64)
        for i in range(D):
            for j in range(N):
                K[i, j] = self.kernel(testset[i], dataset[j], C)

        return K

    def projection_centering(self, K, projection_kernel):
        """
        Centering for projection kernel
        :param K:
        :param projection_kernel:
        :return:
        """
        D, N = projection_kernel.shape
        # one_D = np.ones((D, D), dtype=int) / D
        one_N = np.ones((N, N), dtype=np.float64) / N
        one_DN = np.ones((D, N), dtype=np.float64) / N
        K_centered = projection_kernel - np.dot(one_DN, K) - np.dot(projection_kernel, one_N) + np.dot(one_DN,
                                                                                                       np.dot(K, one_N))
        # K_centered = projection_kernel - np.dot(one_D, projection_kernel) - np.dot(projection_kernel, one_N) + np.dot(one_D, np.dot(projection_kernel, one_N))
        return K_centered

    def projection_contours(self, M, K, V, dataset, I, C, region, plt):
        """
        Create contour lines where projection value is same over a meshgrid within display region.
        Note: display only in 2-Dimensions
        :param V: Vector
        :param dataset: data
        :param I: Vector index
        :param C: sigma
        :param region: region of display
        :param plt: plot contour to
        :return: updated plot
        """

        # create meshgrid
        XX_x, YY_y = np.meshgrid(np.linspace(region[0, 0], region[0, 1], M),
                                 np.linspace(region[1, 0], region[1, 1], M))
        XX = XX_x.reshape([XX_x.shape[0] * XX_x.shape[1], 1])
        YY = YY_y.reshape([YY_y.shape[0] * YY_y.shape[1], 1])
        ZZ = np.column_stack([XX, YY])

        projection_kernel = self.projection_kernel(dataset, ZZ, C)

        projection_matrix_centered = self.projection_centering(K, projection_kernel)

        projection_matrix = projection_matrix_centered * np.matrix(V[:, I]).transpose()

        projection_matrix = projection_matrix.reshape([M, M])
        # IM = plt.imshow(projection_matrix, interpolation='bilinear', origin='lower', cmap=cm.Blues,
        #                extent=(region[0, 0], region[0, 1], region[1, 0], region[1, 1]))
        CS = plt.contourf(XX_x, YY_y, projection_matrix, cmap=cm.Blues, origin='lower')
        CM = plt.contour(CS, levels=CS.levels[::1], linewidths=2, colors='r', hold='on', origin='lower')
        # plot.colorbar(IM)
        # plot.colorbar(CM, orientation='horizontal')

    def thread_func(self, M, K, j, V_n, data_training, region):
        """
        used only for threading
        :return:
        """
        ax1 = plot.subplot('24%d' % (j + 1))
        # plot.figure(j)
        # ax1 = plot.subplot('111')
        self.projection_contours(M, K, V_n, data_training, j, 0.1, region, ax1)
        ax1 = self.plot_data(data_training, ax1)
        plot.xlim(region[0, :])
        plot.ylim(region[1, :])
        plot.title('eigenvalue %.2f' % abs(L[j]))

    def normalized_eigenVectors(self, data_training, C):
        """
        Get normalized eigenvectors
        :return:
        """
        K = self.Kernel_Gram(data_training, C)

        K_c = self.Kernel_Centering(K)

        [L, U] = np.linalg.eig(K_c)
        L = L.real

        sort_index = np.argsort(L, axis=0)
        sort_index = sort_index[::-1]

        print("L>0: " +str(len(np.where(L>0)[0])))
        print("L<0: " + str(len(np.where(L < 0)[0])))
        print("L=0: " +str(len(np.where(L == 0)[0])))
        L = L[sort_index]
        U = U[:, sort_index]

        return K, self.normalize_eigVec(U, L)

    def gamma_weights(self, V, projection_matrix, nComponents):
        """
        Calculate Gamma for all dataset,
        :param V: normalized vector (alpha_i),
        :param projection_matrix: (T x n) Matrix of beta_k values for all test set,
        :param nComponents: number of components in projection,
        :return: Gamma_i = sum_k( alpha_i * beta_k)
        """

        return np.dot(projection_matrix, (V[:, :nComponents]).transpose())


class Gaussian_Kernel(kernelPCA):
    """
    Extending the kernelPCA class with a Gaussian Kernel and Gaussian methods (eq 10 in PCA paper)
    """

    def __init__(self):
        self.name = 'GaussianKernel'
        super(Gaussian_Kernel, self).__init__()

    def kernel(self, x, y, C):
        """
        Return value of Gaussian kernel function
        :param x: vector
        :param y: vector
        :param C: sigma
        :return: value k(x,x') = exp(-||x - x'||² / C)
        """
        return math.exp(-(np.linalg.norm((x - y), ord=2) ** 2) / C)

    # deprecate
    def zKernel(self, z_init):
        """
        Calculate Gaussian kernel k(z_init, dataset)
        :param z_init:
        :return:
        """
        z_kernel = np.zeros((self.nClusters * self.nTestPoints, self.nClusters * self.nTrainingPoints),
                            dtype=np.float64)
        for i in range(self.nClusters * self.nTestPoints):
            for j in range(self.nClusters * self.nTrainingPoints):
                z_kernel[i, j] = self.kernel(z_init[i, :], self.training_data[j, :], self.C)
        return z_kernel

    def approximate_input_data(self, gamma, z_init, training_data, C, nDim):
        """
        Return updated value of approximated input data z_(t+1)
        :param gamma:
        :param z_init:
        :return:
        """
        z_kernel = self.projection_kernel(training_data, z_init, C)  # This is not centered? Should it be?
        z_num = np.dot(np.multiply(gamma, z_kernel), training_data)
        z_den = np.sum(np.multiply(gamma, z_kernel), axis=1)
        if len(np.where(z_den == 0)[0]) > 0:
            print('divide by zero!')
        return np.divide(z_num, np.repeat(np.matrix(z_den).transpose(), nDim, axis=1))


    def approximate_z_single(self,gamma,z_s,training_data,C,nDim):
        z_kernel = self.projection_kernel(training_data, z_s, C)  # This is not centered? Should it be?
        gamma_and_kernel=np.multiply(gamma, z_kernel)
        z_num=np.zeros((1,nDim))
        for i in range(training_data.shape[0]):
            z_num+=gamma_and_kernel[0,i]*training_data[i,:]
        z_den = np.sum(np.multiply(gamma, z_kernel), axis=1)
        if z_den == 0:
            # print('Denominator is zero!')
            raise ValueError
        retVal = np.divide(z_num, z_den)
        return retVal

class Linear_Kernel(kernelPCA):
    """
    Extends kernelPCA class with a linear kernel and linear approximation methods.
    """

    def __init__(self):
        self.name = 'LinearKernel'
        super(Linear_Kernel, self).__init__()

    def kernel(self, x, y, C):
        """
        Return value of Linear kernel function
        :param x: vector
        :param y: vector
        :param C: optional constant
        :return: value k(x,x') = x x'(^T)-c
        """
        return np.dot(x, y.transpose())  # -C

    def approximate_z_single(self, eigvec, X, beta):
        """
        project data (in input space) on the eigenvectors of linear PCA
        :param eigvec: alpha in PCA paper
        :param X: data to be projected
        :param beta: beta in PCA paper
        :return: projected points
        """
        V = np.dot(eigvec.transpose(), X)
        projection = np.zeros((beta.shape[0], V.shape[1]), dtype=np.float64)
        for i in range(beta.shape[0]):
            for j in range(beta.shape[1]):
                projection[i] += beta[i][j] * V[j, :]

        return projection
