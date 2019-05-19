import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def gaussian_noise_add(img,prob_noise,mean, variance):
    height,width = img.shape[:2]
    num_noise_pixels = height * width * prob_noise
    noise_array = np.zeros(256,np.uint)
    noise_mat = np.zeros(shape = (img.shape[0],img.shape[1]),dtype = np.uint8)
    for i in range(0,256):
        fx = 1 / math.sqrt(2 * math.pi *(variance)) * math.exp(- ((i- mean)**2) / (2 * variance))
        noise_array[i] = int(fx * num_noise_pixels +0.5)
        #print(noise_array)
    num_noise_pixels = 0

    for i in range(256):
        num_noise_pixels+=noise_array[i]

    for i in range(0,img.shape[1]):
        for j in range(0,img.shape[0]):
            noise_random = random.randint(0, 99) # randomly decide whether to add noise or not
            if  noise_random / 99 < prob_noise:
                index = random.randint(0, 255)  #randomly decide a noise value to add to noise matrix
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
    return img

if  __name__ == "__main__":
    img = cv2.imread("C:\\Users\\ani49\\OneDrive\\Documents\\GitHub\\homework-3-ani4991\\Lenna.png",0)
    #img = np.ones((100,100),np.uint8)
    #img = img * 100
    mean = 0
    variance = 3
    prob_noise = 0.10

    noise_img = gaussian_noise_add(img,prob_noise,mean,variance)
    test1 = np.histogram(noise_img)
    print(test1)
    #plt.hist(noise_img,bins = 'auto')
    #plt.show()
    cv2.imshow("output",noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
