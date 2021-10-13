import cv2
import numpy as np
from harris import harris
from shi_tomasi import shi_tomasi
from math import ceil


def hough(img):
    lines = cv2.HoughLines(img, 1, np.pi/360, 150)

    # lines = cv2.HoughLinesP(img, 1, np.pi/180, 130, 30, 50)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    slopes = {}
    length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    strong_lines = [lines[0]]
    for line in lines[1:]:
        for strong_line in strong_lines:
            print(strong_line)
            strong_rho, strong_theta = strong_line
            rho, theta = line
            if abs(strong_rho - rho) <= 100 and abs(strong_theta - theta) <= 20:
                strong_lines.append(line)

    for i, line in enumerate(lines):
        for rho, theta in line:
            # for x1, y1, x2, y2 in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + length*(-b))
            y1 = int(y0 + length*(a))
            x2 = int(x0 - length*(-b))
            y2 = int(y0 - length*(a))
            color = (0, 0, 255)
            if x2 - x1 == 0 or y1 < ceil(0.25*img.shape[0]) or x2 < ceil(0.25*img.shape[1]):
                continue
            s = ceil((y2 - y1) / (x2 - x1))
            if s not in slopes:
                slopes[s] = i
            else:
                print(f'{i}, {slopes[s]} are duplicates')
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
            # img = cv2.putText(img, str(
            #     i), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

    print(slopes)

    return img
# cv2.imwrite('houghlines5.jpg',img)


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
    # ret1, binary_image = cv2.threshold(
    #     img, 0, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # binary_image = cv2.adaptiveThreshold(
    #     blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 8)
    kernel = np.ones((2, 2), np.uint8)
    # img = floodfill(img)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=30)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img


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
initial1 = 'initial1.PNG'
initial2 = 'initial2.png'

img = cv2.imread(initial1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Before morph operations', img)

morphed_img = morph(img)
cv2.imshow('After morph operations', morphed_img)

# contour_rect = get_contours(morphed_img)
# print(contour_rect)
# x, y, w, h = contour_rect

# cropped = img[y:y + h, x:x + w]
#cv2.imshow('Cropped', cropped)


# canny_img = cv2.Canny(morphed_img, 500, 550, apertureSize=3)
canny_img = cv2.Canny(img, 200, 250, apertureSize=3)
cv2.imshow('Canny', canny_img)
img = hough(canny_img)
cv2.imshow('Hough', img)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        exit()
