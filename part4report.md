## Part A: Test Dataset
For our testing dataset, we augmented the original videos we used to train the model with four additional videos taken at different angles. We used around 60 images per video taken throughout the duration of each video for a total of roughly 240 new images. 
As an example of the differences between our testing set and our training set, this first image is taken from our training set:
![image](https://user-images.githubusercontent.com/26099766/145692465-4ba91862-425e-492a-b565-62d304968aed.png)
This second image is taken from the additional videos we added to our testing set:
![image](https://user-images.githubusercontent.com/26099766/145692503-ceb2e7f1-a0d7-429c-9b33-f88608ffa0b7.png)

The angle of the board and the lighting in the test image is different from the training image, which causes some pieces to appear differently. This is evident in the confusion matrix for the testing dataset: 
![confusion_matrix](https://user-images.githubusercontent.com/26099766/145692586-58a003b7-4fc6-437f-862c-b3f571725a40.png)

While the model still generally performs well on most pieces, it performs much worse on black queens and white queens compared to our initial test dataset which comprised mostly of images similar to the ones in our training set. That initial confusion matrix can be viewed [here](https://github.com/pdavis322/pdavis8-psoga-CV/blob/main/part3report.md)
