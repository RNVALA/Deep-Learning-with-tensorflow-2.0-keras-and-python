# Deep-Learning-with-tensorflow-2.0-keras-and-python
# Neural Network Overview

This project provides an overview of neural networks, TensorFlow, Keras, and PyTorch.

## Neural Network

A neural network is a machine learning model inspired by the human brain. It consists of interconnected artificial neurons that can recognize patterns, make predictions, and perform various tasks.
![1_bhFifratH9DjKqMBTeQG5A](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/assets/112707550/fdedd05f-21cd-4220-8364-7e8ff1478722)

## TensorFlow

TensorFlow is a powerful and flexible machine learning framework developed by Google. It allows users to build and train neural networks using a dataflow graph. TensorFlow is highly optimized for distributed computing and is known for its scalability, making it suitable for large-scale projects.

## Keras

Keras is a high-level neural network API that runs on top of TensorFlow and other machine learning frameworks. It provides a user-friendly and intuitive interface for building and training neural networks. Keras is widely known for its ease of use, flexibility, and portability across different platforms.

## PyTorch

PyTorch is a deep learning framework developed by Facebook. It offers a dynamic computational graph, which allows for more flexibility in model construction and training. PyTorch provides excellent support for tensor computation, has a clean and pythonic syntax, and is favored by researchers and data scientists who prefer a more dynamic and intuitive approach to building and experimenting with neural networks.

# File:- neurak netwrork and its implementation
## project1 :-MNIST Number Identification using TensorFlow

This project demonstrates how to use TensorFlow to train a neural network on the MNIST dataset for number identification.

## Dataset

The MNIST dataset is a collection of 60,000 training images and 10,000 test images of handwritten digits (0 to 9). Each image is a grayscale image of size 28x28 pixels.
![digits_nn](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/assets/112707550/046d9bb3-a28c-44a8-a3de-8e38893f77af)
## Model Architecture

The neural network model used in this project is a simple feedforward neural network with fully connected layers. The architecture consists of an input layer with 784 neurons (28x28 pixels flattened), a hidden layer with a configurable number of neurons, and an output layer with 10 neurons corresponding to the 10 possible digit classes (0 to 9).
<b> For  Implementation, Please Visit the [neurak netwrork and its implementation](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/neural%20network%20and%20its%20implementatioln.ipynb)

## File:-3.Loss and cost function.ipynb
### Loss and Cost Functions

Loss and cost functions are essential components of deep learning models that quantify the error or discrepancy between the predicted output and the target output. They guide the learning process by providing optimization objectives to minimize the error during training.
- Sparse_Categorical_crossentropy
- binary_crossentropy
- categorical_crossentropy
- mean_absolute_error
- mean_squared_error

<b> For  Implementation, Please Visit the [Loss and cost function.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/3.Loss%20and%20cost%20function.ipynb)

## File:-4.Gradient Descent.ipynb
### What is Gradient descent
- Gradient Descent is an optimization algorithm used in machine learning to minimize the loss function of a model. it us a popular technique used to update the parameter of a model in order to findd the best fit for a given dataset
- The algorithm tries to find the optimal values of the model parameters that minimize the error between the predicted and actual value
![nn](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/assets/112707550/751f7b2b-80c2-416e-9ee9-b8f21a79adbf)

<b> For  Implementation, Please Visit the [4.Gradient Descent.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/4.Gradient%20Descent.ipynb)

## File:-5.stocasticated_descent_batch_Gradient_and_mini_batch_gradient.ipynb
#### What is Batch Gradient Descent
- In Batch gradient descent we go through all training samples and calculate cumulative error
- For example,if i have 10 million samples to find cumulative error for first round(epoch) now we need to do a forward pass for 10 million samples.
#### What is  stochastic Gradient Descent
- instead of going through all sample go to one randomly pick data training sample find out the error.
- and after first sample you start adjusting the weights
1. w1=w1-learning rate*d(error)/d(w1)
2. w2=w2-learning rate*d(error)/d(W2)
3. b=b-learning rate*d(error)/d(bias)
- Again randomly pick a training sample then find out the error,again adjust weights.
- so, you adjust weight after every training sample forward pass this is called stochastic graadient descent.
#### What is mini batch gradient
- Mini Batch is like SGD. Instead of choosing one randomly picked training sample, you will use a batch of randomly picked training samples.
- for example, I have 20 training samples total. Let's say I use 5 random samples for one forward pass to calculate cumulative error
- After that  adjust weights.




  




