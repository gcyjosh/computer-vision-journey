import cv2

image = cv2.imread("cat2.jpg")

if image is None:
    print("Error: Could not print image.")
else:
    cv2.imshow("Cat Image",image)
    height, width, channels = image.shape #shape returns a NumPy array.
    print("Witdth: ", width)
    print("Height: ", height)
    print("Channels: ", channels)

    cv2.waitKey(0)
    cv2.destroyAllWindows()