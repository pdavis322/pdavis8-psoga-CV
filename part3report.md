
## Part B: Classification Accuracy
The following image shows the performance of the model on our validation set in the form of a confusion matrix. 
![confusion_matrix](https://user-images.githubusercontent.com/26099766/142708871-e8fa438e-5fb7-4f2b-b64c-f9ab1595a32a.png)
The following image shows a PR curve of the model on our validation set.
![PR_curve](https://user-images.githubusercontent.com/26099766/142709207-ae4d31e7-758c-4d97-9f38-614edf7ba2a0.png)

## Part C: Observed Accuracy / Ideas for Improvement
Looking at the above confusion matrix, the model performs very well on pieces which are well represented in our training data, like white and black pawns, and less so on pieces which are underrepresented, like white knights or black bishops. In our training data, we have 1,081 instances of pieces labeled as "white pawn" and 817 instances of pieces labeled "black pawn", yet only 182 instances of pieces labeled "white knight". This is caused by the inherent imbalance in the number of pieces in a chess game, and also due to the variations in how long pieces are present throughout the course of a game. In order to increase the accuracy of underrepresented pieces, it would be necessary to increase the frequency of their labels with respect to the overrepresented pieces. However, because pieces like pawns will always be overrepresented, this may be unavoidable.
