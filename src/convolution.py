import cv2
import numpy as np


image = cv2.imread('./../resource/2.jpg')
cv2.namedWindow('src')
cv2.imshow('src', image)
cv2.waitKey(0)
cv2.destroyWindow('src')
kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

dst = cv2.filter2D(image, -1, kernel1)
cv2.namedWindow("dst")
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyWindow('dst')
cv2.imwrite('./../resource/2_conv.jpg', dst)
