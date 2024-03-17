# CS6910 Assignment 01
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

# Link to the wandb project report
https://wandb.ai/ee22m008/CS6910_EE22M008_A1/reports/-CS6910-Assignment-1--Vmlldzo3MTkwMTY0

# Explanation of the project:
This Github repository presents the codes for assignment 1 of CS6910. For ease of uploading and wandb integration, we have uploaded different versions of the code according to the tasks performed.

1. EE22M008_Assignment_01_wandb.ipynb is the code which containes 50 sweeps configuration for the fashion MNIST dataset for both cross entropy and mean squared error asked in question 1, 2, 3, 4. This coe generates wandb plots and can be visualized in wand workspace. This code helps to find the best hyperparameters.

2. EE22M008_Assignment_01.ipynb is used to visualize the performance of the MNIST dataset using the best hyperparameters obtained from wandb sweeps

3. EE22M008_Assignment_01_wandb.ipynb is the code which contains 50 wandb sweeps and simulations for the Fashion MNIST datsaset for both the Mean squared error loss as well as the cross entropy function. Compared both the loss as specified by question 8.

4. EE22M008_Assignment_01.ipynb is the code which trains on the 3 recommended hyperparameter configurations for the MNIST dataset as specified in question 10.

The Neural Networks training framework:

Our code is structured around a procedural framework and does not utilize classes for neural network models, as is often the case in libraries like Keras. This design choice was made to prioritize simplicity and ease of understanding. Our code is tailored specifically for classification tasks and assumes, by default, that the activation function for the last layer is softmax. This simplifies the implementation as the tasks in the assignment do not require a different output layer activation function

1. model_fit()

   This function contains all the optimization techniques with different specifications

activation: activation functions for all the layers except the last layer which is softmax 
 (sigmoid, ReLU, tanh)

weight init_mode: initialization mode 
 (random_normal, xavier)

optimizer: optimization routine 
 (sgd, momentum, nesterov, RMSprop, Adam, nadam)

bach_size: minibatch size

loss: loss function 
 (MSE, Crossentropy)

epochs: number of epochs to be used

L2_lamb: lambda for L2 regularisation of weights

num_neurons: number of neurons in every hidden layer

num_hidden: number of hidden layers

The function returns

parameterss: a dictionary containing all weights and biases. for e.g params["Wi"] is the Weight matrix from i-1 th layer to the ith layer.

A summary of the update stages is given below:
