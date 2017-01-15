import numpy as np
import csv
import matplotlib.pyplot as plt
from kernelPCA_Class import Gaussian_Kernel, Linear_Kernel
from sklearn.decomposition import PCA
import pdb
from random import shuffle
from skimage.util import random_noise


class kPCA_usps():
    def __init__(self):
        self.name = 'usps_example'
        self.data_labels, self.data = self.readData('USPS_dataset//zip.train')
        self.training_images = self.extractDataSets(self.data, self.data_labels, 300)
        self.test_labels, self.test_images = self.readData('USPS_dataset//zip.test')
        # self.test_images = self.extractDataSets(self.test_data, self.test_labels, 50)
        # self.test_images=self.shuffleListAndExtract(self.test_images,50)
        #self.gaussian_images = self.addGaussianNoise(np.copy(self.test_images))
        self.gaussian_images = self.addGaussianNoise_skimage(np.copy(self.test_images), 0, 0.25)
        self.speckle_images = self.addSpeckleNoise(np.copy(self.test_images))
        self.kPCA_gaussian = Gaussian_Kernel()
        self.kPCA_linear=Linear_Kernel()
        self.C = 0.5 * 256
        #self.kGram = None
        #self.norm_vec = None
        self.kGram=np.loadtxt('kGram.txt')
        self.norm_vec=np.loadtxt('normVec.txt')

    def addGaussianNoise_skimage(self, data, mean, var):
        noisy_images = []
        for img in data:
            target = np.reshape(img, [16, 16])
            noisy_img = random_noise(target, mode='gaussian', mean=mean, var=var)
            noisy_images.append(np.reshape(noisy_img, [256,]))
        return np.asarray(noisy_images)

    def readData(self, filePath):
        labels = []
        images = []
        with open(filePath, 'r') as f:
            content = f.readlines()
            for index, pixels in enumerate(content):
                # split string of floats into array
                pixels = pixels.split()
                # the first value is the label
                label = int(float(pixels[0]))
                # the reset contains pixels
                pixels = -1 * np.array(pixels[1:],
                                       dtype=float)  # flips black => white so numbers are black, background white
                # Reshape the array into 16 x 16 array (2-dimensional array)
                labels.append(label)
                images.append(pixels)
        return np.asarray(labels), np.asarray(images)

    def createRandomImage(self):
        return np.asarray(np.random.uniform(-1,1,256)).reshape(1,256)

    def extractDataSets(self, data, labels, nEach):
        # shuffleIndices = np.arange(data.shape[0])
        # shuffle(shuffleIndices)
        # data = data[shuffleIndices, :]
        numbers = 10
        number_label = np.ones((10, 1)) * (nEach - 1)
        retVal = np.zeros((nEach * numbers, data.shape[1]), dtype=float)
        count = 0
        for t, i in enumerate(labels):
            if (number_label[i] < 0):
                continue
            number_label[i] -= 1
            retVal[count, :] = data[t, :]
            count += 1
        return retVal

    def shuffleListAndExtract(self, list, noElements):
        noImages = list.shape[0]
        shuffleIndicies = np.arange(noImages)
        shuffle(shuffleIndicies)
        list = list[shuffleIndicies]
        list = list[:noElements]
        return list

    def addSpeckleNoise(self, images):
        noisy_images = []
        for image in images:
            p = 0.2
            pixels_speckle_noise_push = np.array(
                [np.random.choice([-1., num, 1.], p=[p / 2., 1 - p, p / 2.]) for num in image])
            noisy_images.append(pixels_speckle_noise_push)
        return np.asarray(noisy_images)

    def addGaussianNoise(self, images):
        noisy_images = []
        for row in images:
            pixels = row
            '''add noise here '''
            mu = -1
            sigma = 0.5
            pixels_plus_noise = pixels + np.random.normal(loc=mu, scale=sigma, size=256)
            index = np.where(abs(pixels_plus_noise) > 1)
            pixels_plus_noise[index] = np.sign(pixels_plus_noise[index])
            noisy_images.append(pixels_plus_noise)
        return np.asarray(noisy_images)

    def display(self, images):
        for image in images:
            image = image.reshape((16, 16))
            plt.imshow(image, 'gray', interpolation='none')
            # plt.show()

    def catch_zero(self, gamma, z_init,counter):
        try:
            approx_z = self.kPCA_gaussian.approximate_z_single(gamma, z_init, self.training_images,
                                                               self.C, 256)
        except ValueError as e:
            counter += 1
            reset_z = self.gaussian_images[np.random.choice(self.gaussian_images.shape[0], size=1)]
            approx_z = self.catch_zero(gamma, np.matrix(reset_z),counter)
        return approx_z, counter


    def kernelPCA_gaussian(self, max_eigVec_lst, threshold, test_image,nIteration,test_images):
        # create Projection matrix for all test points and for each max_eigVec
        if self.kGram == None:
            self.kGram, self.norm_vec = self.kPCA_gaussian.normalized_eigenVectors(self.training_images, self.C)
            np.savetxt('kGram.txt', self.kGram)
            np.savetxt('normVec.txt', self.norm_vec)
        projection_kernel = self.kPCA_gaussian.projection_kernel(self.training_images, test_image, self.C)
        projection_matrix_centered = self.kPCA_gaussian.projection_centering(self.kGram, projection_kernel)
        result_lst = []
        for max_eigVec in max_eigVec_lst:
            print(max_eigVec)
            reconstructed_images = []
            projection_matrix = np.dot(projection_matrix_centered, self.norm_vec[:, :max_eigVec])
            # approximate input
            gamma = self.kPCA_gaussian.gamma_weights(self.norm_vec, projection_matrix, max_eigVec)
            print(gamma.shape)
            z_init = np.copy(test_image)  # according to first section under chapter 4,
            max_distance = 10
            count = 0
            while max_distance > threshold and count<nIteration:
                success=0
                while success==0:
                    try:
                        approx_z = self.kPCA_gaussian.approximate_z_single(gamma, z_init, self.training_images,
                                                                           self.C, 256)
                        success=1
                    except ValueError as e:
                        print(e)
                        z_init = test_images[np.random.choice(test_images.shape[0], size=1)]
                max_distance = (np.linalg.norm(z_init - approx_z, axis=1, ord=2))
                z_init = np.copy(approx_z)
                count += 1
            print(max_distance)
            print(count)
            reconstructed_images.append(z_init)
            result_lst.append(reconstructed_images)
        return result_lst

    def kernelPCA_linear(self,max_eigVec_lst):
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


    def findGaussianEigenVectorImages(self, eig_vec_lst,threshold,nIteration):
        # create Projection matrix for all test points and for each max_eigVec
        #if self.kGram == None:
        self.kGram, self.norm_vec = self.kPCA_gaussian.normalized_eigenVectors(self.training_images, self.C)
        result_lst = []
        for eig_vec in eig_vec_lst:
            reconstructed_image = []
            beta=np.zeros((1,self.training_images.shape[0]))
            beta[0,eig_vec]=1
            gamma = np.copy( self.norm_vec[:,eig_vec])
            randomImage=self.createRandomImage()
            z_init=randomImage
            success=0
            max_distance = 10
            count = 0
            while max_distance > threshold and count<nIteration:

                while success==0:
                    try:
                        approx_z = self.kPCA_gaussian.approximate_z_single(gamma, z_init, self.training_images,
                                                                   self.C, 256)
                        success=1
                    except ValueError as e:
                        z_init=self.createRandomImage()

                max_distance = (np.linalg.norm(z_init - approx_z, axis=1, ord=2))
                old_old=np.copy(z_init)
                z_init = np.copy(approx_z)
                count += 1
            print(max_distance)
            print(count)
            reconstructed_image.append(z_init)
            result_lst.append(reconstructed_image)
        return result_lst

    def findLinearEigenVectorImages(self, eig_vec_lst):
        # create Projection matrix for all test points and for each max_eigVec
        self.kGram, self.norm_vec = self.kPCA_linear.normalized_eigenVectors(self.training_images, self.C)
        result_lst = []
        for eig_vec in eig_vec_lst:
            reconstructed_image=np.zeros((1,256))
            for i in range(self.training_images.shape[0]):
                reconstructed_image+=self.norm_vec[i,eig_vec]*self.training_images[i,:]
            result_lst.append(reconstructed_image)
        return result_lst


    def findEigenVectors(self):
        plt.title('Original')
        max_eigVec_lst = [0, 1, 3, 7, 15, 31, 63, 127, 255]
        reconstructed_images = self.findLinearEigenVectorImages(max_eigVec_lst)
        #reconstructed_images=self.findGaussianEigenVectorImages(max_eigVec_lst,10**(-2),1000)
        for i in range(9):
            ax2 = plt.subplot("19%d" % (i + 1))
            self.display(reconstructed_images[i])
            plt.title('comp:%d' % (max_eigVec_lst[i] +1) )
        plt.show()

    def reconstruct(self,nIteration,test_images,digit_loc):
        plt.subplot(1,21,1)
        #self.display(self.gaussian_images[2:3])
        # there is a number "3" in the 2:nd place in the list i.e usps.gaussian_images[2]
        max_eigVec_lst=np.arange(20).reshape(20)+1
        reconstructed_images = self.kernelPCA_gaussian(max_eigVec_lst, 10**(-3), np.matrix(self.test_images[digit_loc]),nIteration,test_images)
        for i in range(20):
            ax2 = plt.subplot(1,21,(i + 1))
            self.display(reconstructed_images[i])
            plt.title('n_comp:%d' % max_eigVec_lst[i])

        ax2=plt.subplot(1,21,21)
        self.display(self.test_images[2:3])
        plt.title('Original image')
        plt.show()


    def searchLocations(self):
        """
        Searches for the first occurance of each digit in the test set.
        :return: Location of the first occurance of each digit in the test tet
        """
        checkIfFound=np.zeros(10)
        locations=np.zeros(10)
        for i in range(len(self.test_labels)):
            label=self.test_labels[i]
            if checkIfFound[label]==0:
                checkIfFound[label]=1
                locations[label]=i
        return locations

    def deNoiseImages(self,image_data,nIteration):
        #numbers 0-9 first location in test set  [5,16,11,2,13,31,1,30,28,0]
        #self.display(self.gaussian_images[30:31])
        locations=[5,16,11,2,13,32,1,30,28,0]
        for digit in range(0,10):
            max_eigVec_lst=[1,4,16,64,256]
            reconstructed_images = self.kernelPCA_gaussian(max_eigVec_lst, 10**(-3), np.matrix(image_data[locations[digit]]),nIteration,image_data)
            for i in range(5):
                index=int(digit + 10 * i + 1)
                ax2 = plt.subplot(5,10,index)
                self.display(reconstructed_images[i])
        plt.show()

if __name__ == "__main__":
    usps = kPCA_usps()
    usps.deNoiseImages()