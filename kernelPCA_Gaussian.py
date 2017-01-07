import numpy as np
import matplotlib.pyplot as plot
import math
import multiprocessing
import matplotlib.cm as cm


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
        dataset = np.zeros((nClusters, nPoints, nDimension), dtype=float)
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

    def Gaussian_Kernel(self, x, y, C):
        """
        Return value of Gaussian kernel function
        :param x: vector
        :param y: vector
        :param C: sigma
        :return: value k(x,x') = exp(-||x - x'||² / C)
        """
        return math.exp(-(np.linalg.norm((x - y), ord=2) ** 2) / C)

    def Gaussian_Kernel_Gram(self, dataset, C):
        """
        Return gaussian kernel gram matrix for given dataset and sigma value
        k(x,x') = exp(-||x - x'||² / C)
        :param dataset: data
        :param C: sigma
        :return: gram matrix K
        """
        N = dataset.shape[0]
        K = np.zeros((N, N), dtype=float)
        XX, YY = np.meshgrid(list(range(N)), list(range(N)))
        XX = XX.reshape([N * N, 1])
        YY = YY.reshape([N * N, 1])
        ZZ = np.column_stack([XX, YY])
        for i in range(ZZ.shape[0]):
            vec = ZZ[i]
            K[vec[0], vec[1]] = self.Gaussian_Kernel(dataset[vec[0]], dataset[vec[1]], C)

        return K

    def Kernel_Centering(self, K):
        """
        Centering the K matrix
        :param K:
        :return: K_centered
        """
        N = K.shape[0]
        one_N = np.ones((N, N), dtype=int) / N
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
        V_norm = np.zeros((V.shape), dtype=float)
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
        K = np.zeros((D, N), dtype=float)
        for i in range(D):
            for j in range(N):
                K[i, j] = self.Gaussian_Kernel(testset[i], dataset[j], C)

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
        one_N = np.ones((N, N), dtype=int) / N
        one_DN = np.ones((D, N), dtype=int) / N
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
        ax1 = self.plot_data(dataset, ax1)
        plot.xlim(region[0, :])
        plot.ylim(region[1, :])
        plot.title('eigenvalue %.2f' % abs(L[j]))

    def normalized_eigenVectors(self, data_training, C):
        """
        Get normalized eigenvectors
        :return:
        """
        K = self.Gaussian_Kernel_Gram(data_training, C)

        K_c = self.Kernel_Centering(K)

        [L, U] = np.linalg.eig(K_c)
        L = L.real

        sort_index = np.argsort(L, axis=0)
        sort_index = sort_index[::-1]

        L = L[sort_index]
        U = U[:, sort_index]

        return K, self.normalize_eigVec(U, L)

if __name__ == "__main__":
    nClusters = 3
    nDim = 2
    nPoints = 100
    C = 0.1
    STD = 0.6  # standard deviation in each cluster
    M = 100  # meshgrid points
    max_eigVec = 8

    region = np.zeros((nDim, 2), dtype=float)
    region[0, :] = np.array([-2, 2], dtype=float)
    region[1, :] = np.array([-1.5, 2.5], dtype=float)

    # mean = np.random.rand(nClusters, nDim)
    # mean[:, 0] = (mean[:,0] - 0.5) / np.ptp(region[0,:], axis=0)
    # mean[:, 1] = (mean[:, 1] - 0.5) / np.ptp(region[1, :], axis=0)
    mean = np.zeros((3, 2), dtype=float)
    mean[0, :] = [-1, -0.5]
    mean[1, :] = [0.0, 1.6]
    mean[2, :] = [1.5, 0.0]

    stdval = STD * np.eye(nDim, dtype=float).reshape([1, nDim, nDim]).repeat(nClusters, axis=0)
    kPCA = kernelPCA()
    dataset = kPCA.create_gaussian_data(mean, stdval, nPoints, nClusters, nDim)

    data = dataset.reshape([nClusters * nPoints, nDim])
    # selected = np.random.permutation(data.shape[0])
    # train_idx, test_idx = selected[:math.floor(data.shape[0] * 0.8)], selected[math.floor(data.shape[0] * 0.8):]
    # data_training, data_test = data[train_idx, :], data[test_idx, :]
    data_training = data

    K, V_n = kPCA.normalized_eigenVectors(data_training, C)

    # only take first 8 eigenvectors and plot contours on a grid where Projections are same

    # For multiprocessing
    # jobs = []
    # for j in range(max_eigVec):
    #     t = multiprocessing.Process(target=kPCA.thread_func, args=(M, K, j, V_n, data_training, region))
    #     jobs.append(t)
    #     t.start()
    #
    # for x in jobs:
    #     x.join()

    for j in range(max_eigVec):
        kPCA.thread_func(M, K, j, V_n, data_training, region)

    plot.show()
