import cv2
import numpy as np
from harris import harris
from shi_tomasi import shi_tomasi
from math import ceil


def hough(img):
    lines = cv2.HoughLines(img,1,np.pi/180,250)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    slopes = {}
    for i, line in enumerate(lines):
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            color = (0, 0, 255) if i != 4 else (0, 255, 0)
            if x2 - x1 != 0:
                s = ceil((y2 - y1) / (x2 - x1))
                if s not in slopes:
                    slopes[s] = i
                else:
                    print(f'{i}, {slopes[s]} are duplicates')
                    cv2.line(img,(x1,y1),(x2,y2),color ,2)
            img = cv2.putText(img, str(i), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    print(slopes)
    
        
    return img
#cv2.imwrite('houghlines5.jpg',img)

def sobel(img):
    sobelx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    return sobelx, sobely, sobelxy

    # cv2.imshow('Sobel X', sobelx)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.imshow('Sobel XY', sobelxy)


def floodfill(img):
    im_floodfill = img.copy()
    h, w = img.shape[: 2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img = img | im_floodfill_inv
    return img


def morph(img):
    ret1, binary_image = cv2.threshold(
        img, 0, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    # binary_image = cv2.adaptiveThreshold(
    #     blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 8)
    # kernel = np.ones((2, 2), np.uint8)
    # img = floodfill(binary_image)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return binary_image


def get_contours(img):
    contours = []
    ret_contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours += ret_contours
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    max_area_contour = max(contours, key=cv2.contourArea)
    # for contour in contours:
    rect = cv2.boundingRect(max_area_contour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # cv2.imshow('After drawing contours', img)
    cv2.rectangle(img, rect, (0, 0, 255))
    cv2.imshow('after drawing contours', img)
    return rect


without_magnus_white_shirt = 'without_magnus_white_shirt.png'
chessboard = 'chessboard.png'
boy_vs_man = 'boy_vs_man.png'
initial_state = 'initial1.PNG'

img = cv2.imread(initial_state)
'''
scale_percent = 50
img = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)), interpolation = cv2.INTER_AREA)
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Before morph operations', img)

morphed_img = morph(gray)
#cv2.imshow('After morph operations', morphed_img)

contour_rect = get_contours(morphed_img)
print(contour_rect)
x, y, w, h = contour_rect

cropped = img[y:y + h, x:x + w]
#cv2.imshow('Cropped', cropped)

canny_img = cv2.Canny(gray,250,300,apertureSize = 3)
cv2.imshow('Canny', canny_img)
img = hough(canny_img)
cv2.imshow('Hough', img)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        exit()
