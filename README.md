CS6910 Assignment 01
----
Instructions to train and evaluate the neural network models:

To train a neural network model for image classification on the Fashion-MNIST dataset, utilizing categorical cross-entropy loss, refer to the notebook titled EE22M008_Assignment_01_wandb.ipynb

a. This notebook is utilized for visualizing images for each class and facilitating model training with various hyperparameters using sweep configuration through wandb.
b. This notebook compares the performance of models trained using both cross-entropy and mean squared error loss functions.
c. wandb assists in identifying which hyperparameters yield the best performance, with the top 3 recommendations being provided 

To train the model using our 3 recommendations for the set of hyperparameters on the MNIST dataset, use the notebook EE22M008_Assignment_01.ipynb. Run all the cells to train the model and obtain evaluation results.
a. Compared the performance and visualized the confusion matrix for the best 3 hyperparameters

To upload the example images from each class and the metrics confusion matrices plots run the entire cells in the  EE22M008_Assignment_01_wandb.ipynb file.

Note: Wherever you need to log to wandb, please remember to change the name of the entity and project in the corresponding line of code. For the ROC, Precision Recall and Confusion Matrix plots, before running it please remove wandb.save and finish line in model_fit.fit(), in order to execute successfully after successfully running hyperparameter sweep config

