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



    for i in range(-10,10):
        #print(i)
        fx = (1. / math.sqrt(2. * math.pi *(variance))) * math.exp(-(i - mean)**2 / (2. * variance))
        noise_array[count] = fx
        count += 1
        #print(noise_array)
    num_noise_pixels = 0

    cdf_g = np.cumsum(noise_array)

    print("noise array=",cdf_g)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_number = np.random.rand()

            count = -10
            for k in cdf_g:

                if random_number <= k:
                    noise_mat[i,j] = count*20 #scalar value 20 is multiplied to make the noise visible
                    #print(noise_mat[i,j])

                    break
                count += 1

    print(np.max(noise_mat), np.min(noise_mat),noise_mat)
    hist1 = np.histogram(noise_mat)
    plt.hist(noise_mat)
    plt.show()

    return noise_mat
    #exit()
    """for i in range(256):
        num_noise_pixels+=noise_array[i]
    print(num_noise_pixels)
    for i in range(0,img.shape[1]):
        for j in range(0,img.shape[0]):
            noise_random = random.randint(0, 99) # randomly decide whether to add noise or not
            if  noise_random / 99 < prob_noise:
                index = random.randint(0, 256)  #randomly decide a noise value to add to noise matrix
                while True:

                    if num_noise_pixels==0:
                        break
                    if noise_array[index] > 0:
                        noise_mat[i][j] = index
                        noise_array[index]-=1
                        num_noise_pixels -= 1

                        break
                    else:
                        index+=1
                        if index==256:
                            index=0
    img = img + noise_mat

    return img"""

if  __name__ == "__main__":
    img = cv2.imread("C:\\Users\\ani49\\OneDrive\\Documents\\GitHub\\homework-3-ani4991\\Lenna.png",0)
    #img = np.ones((100,100),np.uint8)
    #img = img * 100
    print(img)
    mean = 0
    variance = 1.0
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
    cv2.imwrite("img_with_gaussian_noise.png",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
