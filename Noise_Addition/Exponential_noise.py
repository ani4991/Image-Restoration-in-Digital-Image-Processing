import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def gaussian_noise_add(img,prob_noise,mean, variance):
    height,width = img.shape[:2]
    print(height,width)
    num_noise_pixels = height * width * prob_noise
    print(num_noise_pixels)
    noise_array = np.zeros(20, np.float)
    #print(noise_array)
    noise_mat = np.zeros(shape = (img.shape[0],img.shape[1]),dtype = np.float)
    count = 0

    #variance = 10

    a = 1/mean

    print("a=",a)
    for i in range(-10,10):
        #print(i)
        if i >= 0:
            fx = a * math.exp(-(a*i))
        elif i < a:
            fx = 0
        noise_array[count] = fx
        count += 1
        #print(noise_array)
    num_noise_pixels = 0

    cdf_g = np.cumsum(noise_array)

    #print("noise array=",noise_array)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_number = np.random.rand()

            count = -10
            for k in cdf_g:

                if random_number <= k:
                    noise_mat[i,j] = count*19

                    break
                count += 1

    print(np.max(noise_mat), np.min(noise_mat),noise_mat)
    hist1 = np.histogram(noise_mat)
    #plt.hist(noise_mat)
    #plt.show()

    return noise_mat
    #exit()


if  __name__ == "__main__":
    img = cv2.imread("C:\\Users\\ani49\\OneDrive\\Documents\\GitHub\\homework-3-ani4991\\Lenna.png",0)
    #img = np.ones((100,100),np.uint8)
    #img = img * 100
    #print(img)
    mean = 2
    variance = 2.28
    prob_noise = 0.50
    #hist1 = np.histogram(img)
    #plt.hist(img)
    #plt.show()
    noise_mat = gaussian_noise_add(img,prob_noise,mean,variance)
    #print(noise_img)
    print("got noise matrix and waiting for hist")
    #test1 = np.histogram(noise_img)
    #print(test1)
    #plt.hist(noise_img,bins = 'auto')
    #plt.show()
    img = img + noise_mat
    #hist1 = np.histogram(noise_mat)
    #plt.hist(img)
    #plt.show()
    cv2.imwrite("img_with_exponenetial_noise.png",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()