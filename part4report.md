## Part A: Test Dataset

For our testing dataset, we augmented the original videos we used to train the model with four additional videos taken at different angles.
As an example of the differences between our testing set and our training set, this first image is taken from our training set:
![image](https://user-images.githubusercontent.com/26099766/145692465-4ba91862-425e-492a-b565-62d304968aed.png)
This second image is taken from the additional videos we added to our testing set:
![image](https://user-images.githubusercontent.com/26099766/145692503-ceb2e7f1-a0d7-429c-9b33-f88608ffa0b7.png)

The angle of the board and the lighting in the test image is different from the training image, which causes some pieces to appear differently. Additionally, the varied angle means that certain pieces which were occluded in most of the images in the games from our training set are unoccluded. This is evident in the confusion matrix for the testing dataset:
![confusion_matrix](https://user-images.githubusercontent.com/26099766/145692586-58a003b7-4fc6-437f-862c-b3f571725a40.png)

While the model still generally performs well on most pieces, it performs much worse on black queens and white queens compared to our initial test dataset which comprised mostly of images similar to the ones in our training set. That initial confusion matrix can be viewed [here](https://github.com/pdavis322/pdavis8-psoga-CV/blob/main/part3report.md). In order to improve the performance of the model on these games taken at different angles, it would be necessary to include taken at these angles in the training set. This would improve the performance on pieces which were underrepresented in the training set.

While the piece detection and classification is able to detect many pieces, the rank and file classification is not as successful as we would have liked. Most of this is likely due to the fact that so many games have subtly different angles capturing the board state, and so the skew of the grid lines makes a rank-and-file classification difficult. If the games were recorded from a top-down view, it would be much more straightforward since the space between each grid line would be equal. However, this would also be a less interesting project. This is probably the cause of the innacurate FEN strings and final board output. Severe occlusion and small training data also hamper the performance of YOLOv5 leading to missing pieces.

## Part B: Presentation

Our presentation can be viewed here: https://docs.google.com/presentation/d/1GcJY5jhG46cwS6tUYZgUOAMyr-Ycm9ArT-L9O_G0-GQ/edit?usp=sharing

## Part C: Instructions

In order to run our code, clone the repository, and enter the following commands:

```
pip install requirements.txt
python3 driver.py
```
