import cv2

cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) #macOS uses a camera system called AVFOUNDATION. Without it, OpenCV sometimes chooses the wrong backend and fails on Macs.

print(cam.isOpened())

ret, frame = cam.read()

print(ret)

cam.release()