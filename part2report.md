# Part 2 Report

## Data Preprocessing / Feature Extraction
There is no pre-processing done on the data. For feature extraction, we used Canny edge detection and Hough transforms. 

## Feature Extraction Methodology
We used Canny edge detection in order to get the horizontal and vertical lines of the chessboard. Then, we used Hough line transform to get the coordinates of these lines in polar coordinates. Next, we transform the polar coordinates to Cartesian coordinates. After obtaining Cartesian coordinates, we perform a variety of filters on the lines to reduce them to the correct chessboard lines:
1. Filter lines by start/end points, assuming that the chessboard is in the center of the screen
2. Filter lines by angle
3. Segment lines by angle k-means clustering in order to find intersections
4. Find corners of chessboard based on intersections
