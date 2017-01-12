import numpy as np
from kernelPCA_Class import Gaussian_Kernel,Linear_Kernel
import sys
from sklearn.decomposition import PCA

class kPCA_toy:
    """
    Implementation of toy example in paper "Kernel PCA and De-Noising in feature space" by Mika et. al.
    """
    def __init__(self, mean, var, C, nClusters, nTrainingPoints, nTestPoints, nDim):
        self.name = "kPCA_toy_example"
        self.mean = mean
        self.var = var
        self.std = np.power(var, 0.5)
        self.C = C*nDim #see first section under chap 4 k(x,y)=exp(-|x-y|/(cn))
        self.nClusters = nClusters
        self.nTrainingPoints = nTrainingPoints
        self.nTestPoints = nTestPoints
        self.nDim = nDim
        self.kPCA_gaussian = Gaussian_Kernel()
        self.kPCA_linear=Linear_Kernel()
        self.training_data = self.kPCA_gaussian.create_gaussian_data(mean, self.std, nTrainingPoints, nClusters, nDim)\
                                .reshape([nClusters*nTrainingPoints, nDim])
        self.test_data = self.kPCA_gaussian.create_gaussian_data(mean, self.std, nTestPoints, nClusters, nDim)\
                                .reshape([nClusters*nTestPoints, nDim])

    def kernelPCA_sum_gaussian(self, max_eigVec_lst,threshold):
        # create Projection matrix for all test points and for each max_eigVec
        kGram, norm_vec = self.kPCA_gaussian.normalized_eigenVectors(self.training_data, self.C)
        projection_kernel = self.kPCA_gaussian.projection_kernel(self.training_data, self.test_data, self.C)
        projection_matrix_centered = self.kPCA_gaussian.projection_centering(kGram, projection_kernel)
        mean_sqr_sum_lst=[]
        for max_eigVec in max_eigVec_lst:
            print(max_eigVec)
            projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :max_eigVec])

            # approximate input
            gamma = self.kPCA_gaussian.gamma_weights(norm_vec, projection_matrix, max_eigVec)
            #np.random.seed(20)
            #z_init = np.random.rand(self.nClusters * self.nTestPoints, self.nDim)
            z_init=self.test_data #according to first section under chapter 4,
            # in de-noising we can use the test points as starting guess
            z_init_old=np.zeros(z_init.shape)
            max_distance=1
            while max_distance>threshold:
                z_init_old=z_init
                z_init = self.kPCA_gaussian.approximate_input_data(gamma, z_init,self.training_data,self.C,self.nDim)
                max_distance=max(np.linalg.norm(z_init-z_init_old,axis=1,ord=2))
                #print(max_distance)
            # calculate the mean square distance from cluster center of each point
            z_proj = z_init
            mean_sqr_sum = 0
            for i in range(self.nClusters):
                for j in range(self.nTestPoints):
                    mean_sqr_sum += (np.linalg.norm(z_proj[i * self.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)
            mean_sqr_sum_lst.append(mean_sqr_sum)

        return mean_sqr_sum_lst

    def kernelPCA_sum_linear(self,max_eigVec_lst):
        """
        De-noises test data using a Linear Kernel and direct projection
        :param max_eigVec_lst: list of different number of principal components to be used
        :return: list of the mean square distance for each trial using different max_eigVec
        """

        #centering data
        kGram, norm_vec = self.kPCA_linear.normalized_eigenVectors(self.training_data, self.C)
        projection_kernel = self.kPCA_linear.projection_kernel(self.training_data, self.test_data, self.C)
        projection_matrix_centered = self.kPCA_linear.projection_centering(kGram, projection_kernel)


        mean_sqr_sum_lst=[]
        for max_eigVec in max_eigVec_lst:
            projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :max_eigVec])

            z_proj=self.kPCA_linear.approximate_input_data(norm_vec[:, :max_eigVec],self.training_data,projection_matrix)
            mean_sqr_sum = 0
            for i in range(self.nClusters):
                for j in range(self.nTestPoints):
                    mean_sqr_sum += (np.linalg.norm(z_proj[i * self.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)
            mean_sqr_sum_lst.append(mean_sqr_sum)
        return mean_sqr_sum_lst

    def linearPCA(self,max_eigVec_lst):
        mean_sqr_sum_lst=[]

        for max_eigVec in max_eigVec_lst:

            pca = PCA(n_components=max_eigVec)
            pca.fit(self.training_data)
            z_proj=pca.transform(self.test_data)
            mean_sqr_sum = 0
            for i in range(self.nClusters):
                for j in range(self.nTestPoints):
                    mean_sqr_sum += (np.linalg.norm(z_proj[i * self.nTestPoints + j, :] - mean[i, :], ord=2) ** 2)
            mean_sqr_sum_lst.append(mean_sqr_sum)
        return mean_sqr_sum_lst


if __name__ == "__main__":
    std=float(sys.argv[1])
    results=[]
    # create data
    # choose mean uniformly between [-1,1]
    nDimension = 10
    nClusters = 11
    nTrainingPoints = 100
    nTestPoints = 33
    max_eigVec_lst = [1,2,3,4,5,6,7,8,9]
    convergence_threshold=0.1
    mean = np.random.uniform(-1, 1, nClusters*nDimension).reshape([nClusters, nDimension])
    var = std ** 2
    var_matrix = np.zeros((nClusters, nDimension, nDimension), dtype=float)
    # create class object
    for j in range(nClusters):
        var_matrix[j, :, :] = (var * np.eye(nDimension, dtype=float))
    kPCAtoy = kPCA_toy(mean, var_matrix, 2 * var, nClusters, nTrainingPoints, nTestPoints, nDimension)



    sqr_sum_lst_gaussian = kPCAtoy.kernelPCA_sum_gaussian(max_eigVec_lst,convergence_threshold)
    sqr_sum__lst_linear=kPCAtoy.linearPCA(max_eigVec_lst)

    #print("Sum= %f, std=%f, max_eigVec=%d" % (sqr_sum_gaussian, std, max_eigVec))
    #print("Sum= %f, std=%f, max_eigVec=%d" % (sqr_sum_linear, std, max_eigVec))
    #print("ratio: " + str(sqr_sum_linear/sqr_sum_gaussian))
    # linear PCA
    for i in range(len(sqr_sum_lst_gaussian)):
        results.append(sqr_sum__lst_linear[i]/sqr_sum_lst_gaussian[i])
print("Std: " + str(std))

print(sqr_sum_lst_gaussian)
print(sqr_sum__lst_linear)
fileName="std_"+str(std) +".txt"
np.savetxt(fileName, results, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')