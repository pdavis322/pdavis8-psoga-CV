import cv2
import numpy as np
from numpy.lib.function_base import append
from harris import harris
from shi_tomasi import shi_tomasi
from math import ceil, floor
from collections import defaultdict


def segment_by_angle_kmeans(lines, k=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    attempts = 10

    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    labels, _ = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)

    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    print(len(labels))
    print(len(segmented))
    return segmented


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


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
    # lines = cv2.HoughLines(img, 1, np.pi/360, 150,)
    lines = cv2.HoughLines(img, 1, np.pi/360, 150,
                           #    min_theta=np.pi/12, max_theta=np.pi/3,
                           #    min_theta=-1*np.pi/3, max_theta=np.pi / 3
                           )

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

    if not lines.any():
        return img

    strong_lines = [lines[0]]
    main_lines = []

    x_diffs = [lines[0][0]]

    non_intersecting_lines = [lines[0]]

    x_coords = [lines[0][0][0]]

    lines = _fix_negative_rho_in_hesse_normal_form(lines)
    for line in lines:
        rho, theta = line[0]
        x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
        if x1 <= 0 and y1 <= 0 and x2 >= 0 and y2 >= 0:
            if x2 < ceil(0.55 * img.shape[1]) or y2 < ceil(0.55*img.shape[0]):
                continue
        if x1 <= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            if x2 < ceil(0.55 * img.shape[1]) or y2 < ceil(0.55*img.shape[0]):
                continue

        if x2 - x1 == 0:
            continue

        s = (y2 - y1) / (x2 - x1)
        if abs(s) > 10 or abs(s) < 0.01:
            continue

        if -1*np.pi/3 <= theta <= np.pi/3 or (-1*np.pi/3 - np.pi/4) <= theta <= (np.pi/3 + np.pi/4):
            main_lines.append(line)
            continue

        main_lines.append(line)

    for line in main_lines[1:]:
        append_ = True
        for strong_line in strong_lines:
            strong_rho, strong_theta = strong_line[0]
            rho, theta = line[0]
            # if -1*np.pi/3 <= theta <= np.pi/3 or (-1*np.pi/3 - np.pi/4) <= theta <= (np.pi/3 + np.pi/4):
            #     strong_lines.append(line)
            # DIFF IN RHO, THETA
            # if abs(strong_rho - rho) < 4 and abs(strong_theta - theta) < 2:
            #     append_ = False
            #     continue

            # DIFF IN RECT COORDS
            strong_x1, strong_y1, strong_x2, strong_y2 = hough_to_rect(
                strong_rho, strong_theta, length)
            x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
            # print('diff x1: ', abs(strong_x1 - x1))
            # print('diff x2: ', abs(strong_x2 - x2))
            # if abs(strong_x1 - x1) < 2.5 or abs(strong_x2 - x2) < 4:
            # if abs(strong_x2 - x2) < 0.015 * np.sqrt(img.shape[0]**2 + img.shape[1] ** 2):
            #     if abs(strong_x2 - x2) < 0.7 * np.sqrt(img.shape[0]):
            #         append_ = False
            #         continue

            # x_diffs.append(abs(strong_x1 - x1))

            # INTERSECTING
            # if get_intersection_point(strong_rho, strong_theta, rho, theta):
            #     append_ = False
            #     continue

            # CLOSE Xs
        #     strong_x1, strong_y1, strong_x2, strong_y2 = hough_to_rect(
        #         strong_rho, strong_theta, length)
        #     rho, theta = line[0]
        #     x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
        #     for x_coord in x_coords:
        #         print(f"x_coord: {x_coord}")
        #         print(f"x1: {x1}")
        #         if abs(x1-x_coord) < 10:
        #             append_ = False
        #             continue
        #         x_coords.append(x1)
        if append_:
            strong_lines.append(line)
            # strong_lines.insert(0, line)

    # strong_lines = _fix_negative_rho_in_hesse_normal_form(lines)
    segmented = segment_by_angle_kmeans(main_lines)
    orientation_to_lines = {
        'vert': segmented[1],
        'horiz': segmented[0]
    }
    vert_outlier = min(segment_by_angle_kmeans(
        segmented[1], k=2), key=len)[0]
    horiz_outlier = min(segment_by_angle_kmeans(
        segmented[0], k=2), key=len)[0]

    for i, segmented_vert in enumerate(orientation_to_lines['vert']):
        rho, theta = segmented_vert[0]
        if rho == vert_outlier[0][0] and theta == vert_outlier[0][1]:
            orientation_to_lines['vert'] = orientation_to_lines['vert'][:i] + \
                orientation_to_lines['vert'][i+1:]

    for i, segmented_horiz in enumerate(orientation_to_lines['horiz']):
        rho, theta = segmented_horiz[0]
        if rho == horiz_outlier[0][0] and theta == horiz_outlier[0][1]:
            orientation_to_lines['horiz'] = orientation_to_lines['horiz'][:i] + \
                orientation_to_lines['horiz'][i+1:]

    for orientation in orientation_to_lines:
        for segmented_line in orientation_to_lines[orientation]:
            color = (0, 255, 0) if orientation == 'vert' else (0, 0, 255)
            rho, theta = segmented_line[0]
            x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        # cv2.putText(img,
        #             str(round(theta, 2)), (x2//8, y2//8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    intersections = segmented_intersections(
        [orientation_to_lines['vert'], orientation_to_lines['horiz']])
    for intersection in intersections:
        x, y = intersection[0]
        cv2.circle(img, (x, y), radius=1, color=(255, 0, 0), thickness=3)
    return img
# cv2.imwrite('houghlines5.jpg',img)


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
initial_board = 'initial_board.png'
initial1 = 'initial1.PNG'
initial3 = 'initial3.png'
initial4 = 'initial4.png'
initial2 = 'initial2.png'

img = cv2.imread(chessboard)
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
