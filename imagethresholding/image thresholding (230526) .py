### shows the distribution of pixel intensities
### needed for thresholding, equalisation and enhancement, color analysis and segmentation
### count the number of pixels from 0 to 255

import cv2 
import numpy as np #handles arrays and mathematical operations.
import matplotlib.pyplot as plt #Matplotlib’s plotting tools, used for displaying images, graphs and charts
import os #used for file paths, folders, checking files, navigating directories


def Thresholding():
    image = cv2.imread("cat2.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("cat2_gray.jpg",gray) #save gray image

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('# of pixels')
    plt.title("Gray image histogram")
    plt.xlim(0, 250) ### limit to 250 as 250 and above are all white background pixels
    plt.ylim(0, 10000)
    plt.show()

    thresOpt = [cv2.THRESH_BINARY,
                cv2.THRESH_BINARY_INV,
                cv2.THRESH_TOZERO,
                cv2.THRESH_TOZERO_INV,
                cv2.THRESH_TRUNC]
    
    thresNames = ['binary','binaryInv','toZero','toZeroInv','trunc']

    plt.figure()
    plt.subplot(231)
    plt.imshow(gray,cmap = 'gray')

    for i in range(len(thresOpt)):
        plt.subplot(2,3,i+2)
        _,imgThres = cv2.threshold(gray,40,255,thresOpt[i])
        plt.imshow(imgThres,cmap='gray')
        plt.title(thresNames[i])

    plt.show()


if __name__ == '__main__':
    Thresholding()
