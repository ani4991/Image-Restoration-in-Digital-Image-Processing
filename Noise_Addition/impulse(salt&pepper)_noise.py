import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

if  __name__ == "__main__":
    img = cv2.imread("C:\\Users\\ani49\\OneDrive\\Documents\\GitHub\\homework-3-ani4991\\Lenna.png",0)

    #mean = 4
    #variance = 10
    prob_noise = 0.01
    thres = 1 - prob_noise

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            rdn = random.random()
            if rdn < prob_noise:
                img[i][j] = 0
            elif rdn > thres:
                img[i][j] = 255
            else:
                img[i][j] = img[i][j]

    #plt.hist(img, bins='auto')
    #plt.show()
    #print("got noise matrix and waiting for hist")

    cv2.imwrite("img_with_salt&pepper_noise.png",np.uint8(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()