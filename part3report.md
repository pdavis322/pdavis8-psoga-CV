## Part A: Justification of Classifier

For this project, we used Ultralytics' [YOLOv5](https://github.com/ultralytics/yolov5) model. We needed an object detection model for detecting and classifying chess pieces, so we trained YOLOv5 on images of chess pieces from games with different lighting and angles from Tata Steel Chess India Blitz and World Blitz Championships. These games were played in the blitz format, meaning we could get more diverse images of chess pieces since each game is more fast-paced.

We chose YOLOv5 due to its combination of speed and accuracy. According to some benchmarks from [paperswithcode](https://paperswithcode.com/sota/real-time-object-detection-on-coco), YOLO architectures scored the best mAP and frame rates among the benchmarked models.

## Part B: Classification Accuracy

The following image shows the performance of the model on our validation set in the form of a confusion matrix.
![confusion_matrix](https://user-images.githubusercontent.com/26099766/142708871-e8fa438e-5fb7-4f2b-b64c-f9ab1595a32a.png)
The following image shows a PR curve of the model on our validation set.
![PR_curve](https://user-images.githubusercontent.com/26099766/142709207-ae4d31e7-758c-4d97-9f38-614edf7ba2a0.png)

## Part C: Observed Accuracy / Ideas for Improvement

Looking at the above confusion matrix, the model performs very well on pieces which are well represented in our training data, like white and black pawns, and less so on pieces which are underrepresented, like white knights or black bishops. In our training data, we have 1,081 instances of pieces labeled as "white pawn" and 817 instances of pieces labeled "black pawn", yet only 182 instances of pieces labeled "white knight". This is caused by the inherent imbalance in the number of pieces in a chess game, and also due to the variations in how long pieces are present throughout the course of a game. In order to increase the accuracy of underrepresented pieces, it would be necessary to increase the frequency of their labels with respect to the overrepresented pieces. However, because pieces like pawns will always be overrepresented, this may be unavoidable. Adding more data for pieces from a greater variety of games would also be beneficial to the generalizability of the model, although time spent labeling would become an issue seeing as the dataset we used for this project was entirely hand-labeled. While we augmented our data by flipping axes, it would also be beneficial to add more images of pieces with some blur/adjusted lighting in addition to images of chess pieces from diverse angles, environments, and boards.

So far, our pipeline is chessboard detection followed by intersection detection. Then, YOLOv5 runs and detects the chess pieces and their bounding boxes. Comparing the middle coordinate of the bottom edge of each piece bounding box with the positions of the detected intersections, we classify each piece's file and rank before printing a text representation of the board to the user. The code to detect the pieces and converting a board state into FEN string to display a board with is in the `detect` function in `intersections.py`, and the code for classifying each piece's rank and file based on its bounding box is in `get_position` in `intersections.py`.

## Contributions
Both team members worked on annotating the dataset. Peter worked more on training the model, and Patrick worked more on intersection detection. Both team members worked together on connecting the model with the intersection detection and classifying rank and file based on intersections. The code was written in paired programming sessions.  
