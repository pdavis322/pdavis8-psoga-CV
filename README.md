# Part 2 Report

## Data Preprocessing / Feature Extraction
There is no pre-processing done on the data. For feature extraction, we used Canny edge detection and Hough transforms. 

## Feature Extraction Methodology
We used Canny edge detection in order to get the horizontal and vertical lines of the chessboard. Then, we used Hough line transform to get the coordinates of these lines in polar coordinates. Next, we transform the polar coordinates to Cartesian coordinates. After obtaining Cartesian coordinates, we perform a variety of filters on the lines to reduce them to the correct chessboard lines:
1. Filter lines by start/end points, assuming that the chessboard is in the center of the screen
2. Filter lines by angle
3. Segment lines by angle k-means clustering in order to find intersections
4. Find corners of chessboard based on intersections

## Illustrations
Canny edge 
![Canny](report_images/canny.png)

# Part 1 Deliverable
## Part A: Database
Our database will initially consist of twenty videos of classical chess games being played from start to finish, downloaded from YouTube. Links to these videos are provided in `videos_list.txt`. If more training data is needed, we will use similar videos from YouTube. Additionally, we have downloaded a set of images of chess positions ([roboflow](https://public.roboflow.com/object-detection/chess-full), [kaggle](https://www.kaggle.com/tannergi/chess-piece-detection)) which we will use for classifying chess pieces if neccessary.

## Part B: Description
The first problem would be detecting the chessboard and cropping the image to capture only the board and the pieces. Something like a Hough transform or some edge-detection techniques can help with identifying the chessboard and partitioning it into individual squares. We will need to figure out how to accommodate for the skew and angle of the camera aimed at the board so that the program can correctly read the board state. Once the board can be identified and partitioned into squares, we can begin to track pieces.

In order to track pieces, we can take advantage of the fact that every chess game begins in the same configuration. We can identify which sides of the board have pieces on them, and based on their color, update an initial game state with the positions of the pieces on the board. Then, since only one piece can move at any turn, we can use the difference between the current game state and the previous game state to deduce which piece moved and where the piece was moved to. If need be, we can also use some kind of classification techniques to recognize pieces to inform the game state, whether that be via CNNs or traditional machine learning algorithms with hand-crafted features tailored to each piece. This would give our program the advantage of being able to record chess games starting from any state, not just start to finish. Also, moves like castling would become easier to track since the king and rook can merely swap places.

Often, chess players keep their arms/hands in the chessboard over the pieces. To avoid recording bad game states, we can reject processing frames depending on how many squares are visible/how many edges the cropped chessboard has.
