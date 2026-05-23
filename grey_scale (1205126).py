# Importance of GrayScale:
# GrayScale reduces the data as it only has 1 channel compared to RGB or BGR with 3 channels, improving the efficiency of prepocessing


import cv2

image = cv2.imread("cat2.jpg")

if image is None:
    print("Error: Could not print image.")
else:
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    cv2.imwrite("cat2_gray.jpg",gray) #save gray image

    cv2.imshow("Grayscale Image", gray) #display gray image

    cv2.waitKey(0)

    cv2.destroyAllWindows()
    
