import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('chessboard.png')


def shi_tomasi(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 200, 0.2, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    return img

# plt.imshow(img),plt.show()