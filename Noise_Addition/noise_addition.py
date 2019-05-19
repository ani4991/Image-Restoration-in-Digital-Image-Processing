import cv2
import numpy as np
import random
import math
probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

mean = 0
std_dev = 1
lower_range = 3 *(mean - std_dev)
higher_range = 3 * (mean + std_dev)
img = cv2.imread("link to image")
noise_mat = np.zeros(img.shape[0],img.shaep[1])
ctr =0
pdf = []
cdf = 0
b = 4 / (4 - math.pi)
a = math.sqrt((math.pi * b) /4)
a = -a
for  i in range(0,img.shape[1]):
    for j in range(0,img.shape[0]):
        prob = random.choice(probabilities)
        for k in range(0,100):
            if k >= a:
                pdf[k] = (2 / b) * (k - a) * math.exp((-(k-a)**2)/b)
            else:
                pdf[k]=0
        for s in range(0,100):
            if cdf == prob:
                break
            else:
                cdf += pdf[s]
        noise_mat[i][j] = cdf



