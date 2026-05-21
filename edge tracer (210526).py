import cv2

img = cv2.imread("cat2.jpg")

img_edge = cv2.Canny(img,100,200)

cv2.imshow('img',img)
cv2.imshow('img_edge',img_edge)
cv2.waitKey(0)