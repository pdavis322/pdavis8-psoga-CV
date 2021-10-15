import cv2
import numpy as np
from math import ceil
from collections import defaultdict
from argparse import ArgumentParser


# Segment_by_angle_kmeans, intersection, and segmented_intersections with assistance from Stack Overflow
def segment_by_angle_kmeans(lines, k=2):
    # Settings from OpenCV docs; default k=2 for horizontal and vertical clustering
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
    return segmented


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    # Solve the matrix equation for [x, y]
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


def filter_lines(img, lines, length):
    strong_lines = [lines[0]]
    filtered_lines = []

    for line in lines:
        rho, theta = line[0]
        x1, y1, x2, y2 = hough_to_rect(rho, theta, length)

        # Filter lines by their start and end points assuming that the chessboard
        #   is in the bottom center of the screen
        if x1 <= 0 and y1 <= 0 and x2 >= 0 and y2 >= 0:
            if x2 < ceil(0.55 * img.shape[1]) or y2 < ceil(0.55*img.shape[0]):
                continue
        if x1 <= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            if x2 < ceil(0.55 * img.shape[1]) or y2 < ceil(0.55*img.shape[0]):
                continue

        if x2 - x1 == 0:
            continue

        s = (y2 - y1) / (x2 - x1)

        # Get rid of lines that are too vertical or horizontal
        # Need to find a better method since now we can't mark chessboards
        #   head-on
        if abs(s) > 10 or abs(s) < 0.01:
            continue

        # Only accept lines within a certain angle range; problems similar to those above
        if -1*np.pi/3 <= theta <= np.pi/3 or (-1*np.pi/3 - np.pi/4) <= theta <= (np.pi/3 + np.pi/4):
            filtered_lines.append(line)
            continue

        filtered_lines.append(line)

    for line in filtered_lines[1:]:
        append_ = True
        for strong_line in strong_lines:
            strong_rho, strong_theta = strong_line[0]
            rho, theta = line[0]

            # Filter by difference in x-coordinates to get rid of
            #   lines that are too close together; needs work
            strong_x1, strong_y1, strong_x2, strong_y2 = hough_to_rect(
                strong_rho, strong_theta, length)
            x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
            if abs(strong_x2 - x2) < 0.015 * np.sqrt(img.shape[0]**2 + img.shape[1] ** 2):
                if abs(strong_x2 - x2) < 0.7 * np.sqrt(img.shape[0]):
                    append_ = False
                    continue

        if append_:
            strong_lines.append(line)
    return filtered_lines


def process_lines(lines):
    segmented = segment_by_angle_kmeans(lines)
    orientation_to_lines = {
        'vert': segmented[1],
        'horiz': segmented[0]
    }

    # Get rid of outlier lines within each cluster
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

    return orientation_to_lines


def get_corners(orientation_to_lines, length):

    def by_midpoint(orientation='vert'):
        def calc(line):
            x_temp = line.reshape(-1)
            x1, y1, x2, y2 = hough_to_rect(x_temp[0], x_temp[1], length)
            if orientation == 'vert':
                return (x1 + x2) // 2
            return (y1 + y2) // 2

        return calc

    # Left intersection points = intersection of smallest-x vertical with the extreme horizontal
    #   lines; right intersection points are symmetric
    left = segmented_intersections(
        [[min(orientation_to_lines['vert'], key=by_midpoint('vert'))], [
            max(orientation_to_lines['horiz'], key=by_midpoint('horiz')),
            min(orientation_to_lines['horiz'], key=by_midpoint('horiz'))]
         ])
    right = segmented_intersections(
        [[max(orientation_to_lines['vert'], key=by_midpoint('vert'))], [
            min(orientation_to_lines['horiz'], key=by_midpoint('horiz')),
            max(orientation_to_lines['horiz'], key=by_midpoint('horiz'))]
         ])

    return left + right


def draw_lines(img, lines, length):
    filtered_line_img = img
    for line in lines:
        rho, theta = line[0]
        x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
        cv2.line(filtered_line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)


def draw_corners(img, corners):
    for corner in corners:
        x, y = corner[0]
        cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=3)


def draw_segmented_lines(img, orientation_to_lines, length):
    for orientation in orientation_to_lines:
        for segmented_line in orientation_to_lines[orientation]:
            color = (0, 255, 0) if orientation == 'vert' else (0, 0, 255)
            rho, theta = segmented_line[0]
            x1, y1, x2, y2 = hough_to_rect(rho, theta, length)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)


def main(args):
    file_path = args.file
    output_path = args.output if args.output else None

    img = cv2.imread(file_path)
    cv2.imshow('Initial', img)

    img = cv2.Canny(img, 200, 250, apertureSize=3)
    cv2.imshow('Canny', img)

    length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

    # Hough transform
    lines = cv2.HoughLines(img, 1, np.pi/360, 150)
    if not lines.any():
        print('No lines detected')
        return
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lines_img = img.copy()
    draw_lines(lines_img, lines, length)
    cv2.imshow('Initial lines', lines_img)

    # Line processing
    filtered_lines = filter_lines(img, lines, length)
    filtered_line_img = img.copy()
    draw_lines(filtered_line_img, filtered_lines, length)
    cv2.imshow('Filtered lines', filtered_line_img)

    orientation_to_lines = process_lines(filtered_lines)
    segmented_lines_img = img.copy()
    draw_segmented_lines(segmented_lines_img, orientation_to_lines, length)
    cv2.imshow('Lines segmented by orientation', segmented_lines_img)

    # Corner processing
    corners = get_corners(orientation_to_lines, length)
    draw_corners(img, corners)

    # Final drawing
    draw_segmented_lines(img, orientation_to_lines, length)
    if output_path:
        print(output_path)
        cv2.imwrite(output_path, img)

    cv2.imshow('After processing', img)

    while True:
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            exit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, help='Chessboard file to process')
    parser.add_argument('--output', type=str,
                        help='Output file location')
    args = parser.parse_args()
    main(args)
