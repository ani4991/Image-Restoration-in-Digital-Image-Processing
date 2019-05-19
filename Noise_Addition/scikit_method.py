import skimage.util as sk
import cv2
import numpy as np

def noise(type,img):
    if type == "1":

        print("original matrix",img)
        noise_img = sk.random_noise(img,mode='gaussian', seed=None,clip=False)

        print("matrix values",noise_img)
        return noise_img

    elif type == "2":

        print("original matrix", img)
        noise_img = sk.random_noise(img, mode='s&p', seed=None, clip=False)
        print("matrix values", noise_img)
        return noise_img

    elif type == '4':

        print("original matrix", img)
        noise_img = sk.random_noise(img, mode='speckle', seed=None, clip=False)
        print("matrix values", noise_img)
        return noise_img

    elif type == '3':

        print("original matrix", img)
        noise_img = sk.random_noise(img, mode='poisson', seed=None, clip=False)
        print("matrix values", noise_img)
        return noise_img

    else:

        noise_img = sk.random_noise(img, mode='gaussian', seed=None,clip=False)
        return noise_img


if  __name__ == "__main__":
    img = cv2.imread("C:\\Users\\ani49\\OneDrive\\Documents\\GitHub\\homework-3-ani4991\\Lenna.png",0)
    print("type of noise:")
    print("\t 1. Gaussian \n\t 2. salt&pepper \n\t 3.Poisson \n\t 4.speckle")
    type = input("enter your choice \n")
    noisy_img = noise(type, img)
    cv2.imshow("img",noisy_img)
    cv2.imwrite(type + "_noise_img.jpg", noisy_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()