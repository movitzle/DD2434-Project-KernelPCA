import numpy as np
from kernelPCA_Class import Gaussian_Kernel, Linear_Kernel
import sys
from sklearn.decomposition import PCA, KernelPCA


class kPCA_toy:
    """
    Implementation of toy example in paper "Kernel PCA and De-Noising in feature space" by Mika et. al.
    """

    def __init__(self, mean, var, C, nClusters, nTrainingPoints, nTestPoints, nDim):
        self.name = "kPCA_toy_example"
        self.mean = mean
        self.var = var
        self.std = np.power(var, 0.5)
        self.C = C * nDim  # see first section under chap 4 k(x,y)=exp(-|x-y|/(cn))
        self.nClusters = nClusters
        self.nTrainingPoints = nTrainingPoints
        self.nTestPoints = nTestPoints
        self.nDim = nDim
        self.kPCA_gaussian = Gaussian_Kernel()
        self.kPCA_linear = Linear_Kernel()
        self.training_data = self.kPCA_gaussian.create_gaussian_data(mean, self.std, nTrainingPoints, nClusters, nDim) \
            .reshape([nClusters * nTrainingPoints, nDim])
        self.test_data = self.kPCA_gaussian.create_gaussian_data(mean, self.std, nTestPoints, nClusters, nDim) \
            .reshape([nClusters * nTestPoints, nDim])

    def kernelPCA_sum_gaussian(self, max_eigVec_lst, threshold):
        # create Projection matrix for all test points and for each max_eigVec
        kGram, norm_vec = self.kPCA_gaussian.normalized_eigenVectors(self.training_data, self.C)
        projection_kernel = self.kPCA_gaussian.projection_kernel(self.training_data, self.test_data, self.C)
        projection_matrix_centered = self.kPCA_gaussian.projection_centering(kGram, projection_kernel)
        mean_sqr_sum_lst = []
        for max_eigVec in max_eigVec_lst:
            print(max_eigVec)
            projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :max_eigVec])

            # approximate input
            gamma = self.kPCA_gaussian.gamma_weights(norm_vec, projection_matrix, max_eigVec)
            # np.random.seed(20)
            # z_init = np.random.rand(self.nClusters * self.nTestPoints, self.nDim)
            z_init = self.test_data  # according to first section under chapter 4,
            # in de-noising we can use the test points as starting guess
            z_init_old = np.zeros(z_init.shape)
            max_distance = 1
            while max_distance > threshold:
                z_init_old = z_init
                z_init = self.kPCA_gaussian.approximate_input_data(gamma, z_init, self.training_data, self.C, self.nDim)
                max_distance = max(np.linalg.norm(z_init - z_init_old, axis=1, ord=2))
                # print(max_distance)
            # calculate the mean square distance from cluster center of each point
            z_proj = z_init
            mean_sqr_sum = 0
            for i in range(self.nClusters):
                for j in range(self.nTestPoints):
                    mean_sqr_sum += (np.linalg.norm(z_proj[i * self.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)
            mean_sqr_sum_lst.append(mean_sqr_sum)

        return mean_sqr_sum_lst


    def kernelPCA_sum_gaussian_single(self, max_eigVec_lst, threshold):
        # create Projection matrix for all test points and for each max_eigVec
        kGram, norm_vec = self.kPCA_gaussian.normalized_eigenVectors(self.training_data, self.C)
        projection_kernel = self.kPCA_gaussian.projection_kernel(self.training_data, self.test_data, self.C)
        projection_matrix_centered = self.kPCA_gaussian.projection_centering(kGram, projection_kernel)
        mean_sqr_sum_lst = []
        for max_eigVec in max_eigVec_lst:
            print(max_eigVec)
            projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :max_eigVec])

            # approximate input
            gamma = self.kPCA_gaussian.gamma_weights(norm_vec, projection_matrix, max_eigVec)
            # np.random.seed(20)
            # z_init = np.random.rand(self.nClusters * self.nTestPoints, self.nDim)
            z_init = self.test_data  # according to first section under chapter 4,
            # in de-noising we can use the test points as starting guess
            z_init_old = np.zeros(z_init.shape)
            for tp in range(self.nTestPoints*self.nClusters):
                max_distance = 1
                while max_distance > threshold:
                    try:
                        approx_z = self.kPCA_gaussian.approximate_z_single(gamma[tp, :], np.matrix(z_init[tp, :]), self.training_data,
                                                                           self.C, self.nDim)
                    except ValueError:
                        approx_z = self.kPCA_gaussian.approximate_z_single(gamma[tp, :], np.matrix(z_init[tp-1, :]), self.training_data,
                                                                           self.C, self.nDim)
                    max_distance = (np.linalg.norm(z_init[tp, :] - approx_z, axis=1, ord=2))

                    z_init[tp, :] = approx_z
                    # print(max_distance)
                # calculate the mean square distance from cluster center of each point
            z_proj = z_init
            mean_sqr_sum_lst.append(self.calc_mean_sqr_sum(z_proj))

        return mean_sqr_sum_lst

    def kernelPCA_sum_linear(self, max_eigVec_lst):
        """
        De-noises test data using a Linear Kernel and direct projection
        :param max_eigVec_lst: list of different number of principal components to be used
        :return: list of the mean square distance for each trial using different max_eigVec
        """

        # centering data
        kGram, norm_vec = self.kPCA_linear.normalized_eigenVectors(self.training_data, self.C)
        projection_kernel = self.kPCA_linear.projection_kernel(self.training_data, self.test_data, self.C)
        projection_matrix_centered = self.kPCA_linear.projection_centering(kGram, projection_kernel)

        mean_sqr_sum_lst = []
        # print(norm_vec.shape)
        for max_eigVec in max_eigVec_lst:
            projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :max_eigVec])

            z_proj = self.kPCA_linear.approximate_input_data(norm_vec[:, :max_eigVec], self.training_data,
                                                            projection_matrix)

            mean_sqr_sum_lst.append(self.calc_mean_sqr_sum(z_proj))

        return mean_sqr_sum_lst

    def calc_mean_sqr_sum(self, z_proj):
        mean_sqr_sum = 0
        for i in range(self.nClusters):
            for j in range(self.nTestPoints):
                mean_sqr_sum += (np.linalg.norm(z_proj[i * self.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)
        return mean_sqr_sum

    def linearPCA(self, max_eigVec_lst):
        mean_sqr_sum_lst = []
        mean_training_data = np.mean(self.training_data, axis=0)
        C = np.cov((self.training_data - mean_training_data).transpose())
        L, V = np.linalg.eig(C)
        for max_eigVec in max_eigVec_lst:
            beta = np.dot(self.test_data, V[:, :max_eigVec])
            mean_sqr_sum_tmp = 0
            for i in range(self.nClusters):
                for j in range(self.nTestPoints):
                    projected = np.dot(V[:, :max_eigVec], beta[i * self.nTestPoints + j, :].transpose())
                    mean_sqr_sum_tmp += (np.linalg.norm(projected - mean[i, :], ord=2) ** 2)
            mean_sqr_sum_lst.append(mean_sqr_sum_tmp)
        return mean_sqr_sum_lst

    def scikit_kpca(self, max_eigVec):
        kpca = KernelPCA(kernel='rbf', n_components=max_eigVec, fit_inverse_transform=True, gamma=self.C)
        kpca.fit(self.training_data)
        x_invkpca = kpca.fit_transform(self.test_data)
        x_inv = kpca.inverse_transform(x_invkpca)
        return x_inv

    def scikit_lpca(self, max_eigVec):
        lpca = PCA(n_components=max_eigVec)
        lpca.fit(self.training_data)
        x_invlpca = lpca.fit_transform(self.test_data)
        x_inv = lpca.inverse_transform(x_invlpca)
        return x_inv

if __name__ == "__main__":
    # std=float(sys.argv[1])
    std = 0.05
    results = []
    # create data
    # choose mean uniformly between [-1,1]
    nDimension = 10
    nClusters = 11
    nTrainingPoints = 100
    nTestPoints = 33
    max_eigVec_lst = [1,10,100]
    # max_eigVec_lst = [10]
    convergence_threshold = 0.01
    mean = np.random.uniform(-1, 1, nClusters * nDimension).reshape([nClusters, nDimension])
    var = std ** 2
    var_matrix = np.zeros((nClusters, nDimension, nDimension), dtype=np.float64)
    # create class object
    for j in range(nClusters):
        var_matrix[j, :, :] = (var * np.eye(nDimension, dtype=np.float64))
    kPCAtoy = kPCA_toy(mean, var_matrix, 2 * var, nClusters, nTrainingPoints, nTestPoints, nDimension)

    sqr_sum_lst_gaussian = kPCAtoy.kernelPCA_sum_gaussian_single(max_eigVec_lst, convergence_threshold)
    print("Our kpca %f" % sqr_sum_lst_gaussian[0])
    x_inv = kPCAtoy.scikit_kpca(max_eigVec_lst[0])
    mean_sqr_sum_tmp = 0
    for i in range(kPCAtoy.nClusters):
        for j in range(kPCAtoy.nTestPoints):
            mean_sqr_sum_tmp += (np.linalg.norm(x_inv[i * kPCAtoy.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)

    sqr_sum_lst_scikit = [mean_sqr_sum_tmp]
    print("SCIKIT kpca: %f" % sqr_sum_lst_scikit[0])

    #x_inv = kPCAtoy.scikit_lpca(max_eigVec_lst[0])
    #mean_sqr_sum_tmp = 0
    #for i in range(kPCAtoy.nClusters):
    #    for j in range(kPCAtoy.nTestPoints):
    #        mean_sqr_sum_tmp += (np.linalg.norm(x_inv[i * kPCAtoy.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)

    #sqr_sum__lst_sickit_linear = [mean_sqr_sum_tmp]
    #print("SCIKI LPCA %f" % sqr_sum__lst_sickit_linear[0])

    #sqr_sum__lst_linear = kPCAtoy.linearPCA(max_eigVec_lst)
    #print("OUR LPCA %f" % sqr_sum__lst_linear[0])
    # max_eigVec_lst = [400]
    # sqr_sum_linear_2 = kPCAtoy.kernelPCA_sum_linear(max_eigVec_lst)
    # print(sqr_sum_linear_2)
    # print("Sum= %f, std=%f, max_eigVec=%d" % (sqr_sum_gaussian, std, max_eigVec))
    # print("Sum= %f, std=%f, max_eigVec=%d" % (sqr_sum_linear, std, max_eigVec))
    # print("ratio: " + str(sqr_sum_linear/sqr_sum_gaussian))
    # linear PCA
    for i in range(len(sqr_sum_lst_gaussian)):
        #results.append(sqr_sum__lst_linear[i] / sqr_sum_lst_gaussian[i])
        pass

#print("Std: " + str(std))

print(sqr_sum_lst_gaussian)
#print(sqr_sum_lst_scikit)
#print(sqr_sum__lst_linear[0] / sqr_sum_lst_gaussian[0])
fileName = "std_" + str(std) + ".txt"
np.savetxt(fileName, results, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
