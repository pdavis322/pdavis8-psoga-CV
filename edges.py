import cv2
import numpy as np
from numpy.lib.function_base import append
from harris import harris
from shi_tomasi import shi_tomasi
from math import ceil


def _fix_negative_rho_in_hesse_normal_form(lines):
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = - \
        lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] =  \
        lines[neg_rho_mask, 1] - np.pi
    return lines


def hough_to_rect(rho, theta, length):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + length*(-b))
    y1 = int(y0 + length*(a))
    x2 = int(x0 - length*(-b))
    y2 = int(y0 - length*(a))

    return x1, y1, x2, y2


def get_intersection_point(rho1, theta1, rho2, theta2):
    """Obtain the intersection point of two lines in Hough space.

    This method can be batched

    Args:
        rho1 (np.ndarray): first line's rho
        theta1 (np.ndarray): first line's theta
        rho2 (np.ndarray): second lines's rho
        theta2 (np.ndarray): second line's theta

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: the x and y coordinates of the intersection point(s)
    """
    # rho1 = x cos(theta1) + y sin(theta1)
    # rho2 = x cos(theta2) + y sin(theta2)
    cos_t1 = np.cos(theta1)
    cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1)
    sin_t2 = np.sin(theta2)
    try:
        x = (sin_t1 * rho2 - sin_t2 * rho1) / \
            (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    except ZeroDivisionError:
        x = 0
    try:
        y = (cos_t1 * rho2 - cos_t2 * rho1) / \
            (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    except ZeroDivisionError:
        y = 0
    return x, y


def hough(img):
    lines = cv2.HoughLines(img, 1, np.pi/360, 150,)
    # lines = cv2.HoughLines(img, 1, np.pi/360, 160,
    #                        min_theta=np.pi/12, max_theta=np.pi/3
    #                        )

    # lines = cv2.HoughLinesP(img, 1, np.pi/180, 130, 30, 50)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    slopes = {}
    length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

    strong_lines = [lines[0]]

    x_diffs = [lines[0][0]]

    non_intersecting_lines = [lines[0]]

    x_coords = [lines[0][0][0]]

    for line in lines[1:]:
        append_ = True
        for strong_line in strong_lines:
            strong_rho, strong_theta = strong_line[0]
            rho, theta = line[0]
        # DIFF IN RHO, THETA
        # if abs(strong_rho - rho) < 4 and abs(strong_theta - theta) < 2:
        #     append_ = False
        #     continue

        # DIFF IN RECT COORDS
        # strong_x1, strong_y1, strong_x2, strong_y2 = hough_to_rect(
        #     strong_rho, strong_theta, length)
        # x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
        # if abs(strong_x1 - x1) < 2.5 or abs(strong_x2 - x2) < 4:
        #     append_ = False
        #     continue
        # x_diffs.append(abs(strong_x1 - x1))

        # INTERSECTING
        # if abs(strong_theta) < np.pi/6:
        #     continue
        # if get_intersection_point(strong_rho, strong_theta, rho, theta):
        #     append_ = False
        #     continue

        # CLOSE Xs
        # strong_x1, strong_y1, strong_x2, strong_y2 = hough_to_rect(
        #     strong_rho, strong_theta, length)
        # rho, theta = line[0]
        # x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
        # for x_coord in x_coords:
        #     print(f"x_coord: {x_coord}")
        #     print(f"x1: {x1}")
        #     if abs(x1-x_coord) < 10:
        #         append_ = False
        #         continue
        #     x_coords.append(x1)
        if append_:
            strong_lines.append(line)
            # strong_lines.insert(0, line)

    strong_lines = _fix_negative_rho_in_hesse_normal_form(lines)
    for i, line in enumerate(strong_lines):
        # for rho, theta in line:
        for l in line:
            rho, theta = l
            # for x1, y1, x2, y2 in line:
            x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
            color = (0, 0, 255)
            if x1 - x2 == 0 or x2 < ceil(0.25*img.shape[1]) or y1 < ceil(0.25*img.shape[0]) or x2 < ceil(0.25*img.shape[1]):
                continue
            s = ceil((y2 - y1) / (x2 - x1))
            if s not in slopes:
                slopes[s] = i
            else:
                # print(f'{i}, {slopes[s]} are duplicates')
                cv2.line(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img,
                            str(round(theta, 2)), (x2//4, y2//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    kernel = np.ones((5, 5), np.uint8)
    # img = floodfill(img)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
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
cv2.imshow('Before morph operations', img)

morphed_img = morph(img)
cv2.imshow('After morph operations', morphed_img)

canny_img = cv2.Canny(morphed_img, 200, 250, apertureSize=3)
# canny_img = cv2.Canny(img, 200, 250, apertureSize=3)
cv2.imshow('Canny', canny_img)
img = hough(canny_img)
cv2.imshow('Hough', img)
# img = cv2.resize(img, (640, 480))
# img = shi_tomasi(img)
# cv2.imshow('Corners', img)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        exit()
