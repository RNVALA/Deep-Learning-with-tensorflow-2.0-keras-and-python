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

# File:- neural netwrork and its implementation
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

<b> For  Implementation, Please Visit the [5.stocasticated_descent_batch_Gradient_and_mini_batch_gradient.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/5.stocasticated_descent_batch_Gradient_and_mini_batch_gradient.ipynb) </b>

## File:-6_image_c;assification_and_understand_the_importance_of_gpu.ipynb
### Project:- Image Classification using Artificial neural network and understanding the importance of gpu
#### What is Artificial Neural Network
- An artificial neural network (ANN) is a computational model that is inspired by the structure and function of biological neural networks, such as the human brain. It is composed of a large number of interconnected processing nodes, called neurons, which are organized into layers.
- Each neuron in the network receives input from other neurons in the previous layer, processes the information, and then passes the output to the next layer of neurons. The output of the final layer is the network's prediction or decision based on the input data.
- ANNs are typically used for tasks such as classification, regression, and pattern recognition. They are capable of learning from data by adjusting the strength of the connections between neurons, a process called training or learning. This allows ANNs to improve their performance on a specific task as they receive more training data.

image classification using Artificial Neural Networks (ANN) on the CIFAR-10 dataset.
CIFAR-10 is a popular benchmark dataset consisting of 60,000 32x32 color images across 10 classes.

![small_images](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/assets/112707550/b8895184-42a8-4d44-8d22-7ed8f2edf55c)

Here,50000 images we are using as train set and 10000 images are used as a test set.
step for classification

- Install required libraries
- Load the CIFAR-10 dataset and examine its shape.
- Visualize sample images from the dataset.
- preprocess the data by scaling the pixel to the range 0 to 1
- convert the data to categorical format using keras onehot encoding api
- Define the ANN model architecture
- compile the model by specifying the optimizer,loss function and evaluate the metric.
- Train the model on the trainng data
- make predictioon on the test set and evaluate the model's performance.
<b> For  Implementation, Please Visit the [6_image_c;assification_and_understand_the_importance_of_gpu.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/6_image_c%3Bassification_and_understand_the_importance_of_gpu.ipynb)</b>

## File:-7_Artificial Neural network Customer churn prediction.ipynb
### Project:-Customer Churn Prediction using Artificial neural network
Customer churn refers to the phenomenon where customers stop using the services of a company. Predicting customer churn is crucial for businesses to identify potential at-risk customers and take proactive measures to retain them. This project aims to build a machine learning model that predicts customer churn in a telecommunications company based on historical customer data.

The dataset used for this project contains various features such as customer recharge plans, usage patterns, and service details. By analyzing this data and training a predictive model, we can predict whether a customer is likely to churn or not.


<b> For  detailed Implementation, Please Visit the [7_Artificial Neural network Customer churn prediction.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/7_Artificial%20Neural%20network%20Customer%20churn%20prediction.ipynb)</b>


## File:-8_Dropout Regularization.ipynb
### Project:-The task is to train a network to discriminate between sonar signals bounces off a metal cylinder and those bounced off a roughly cylindrical rock
This project aims to classify sonar signals as either rocks (R) or mines (M) using a neural network model. The dataset used for training and testing the model is the Sonar Dataset.

#### What is Dropout?
Dropout is a regularization technique that helps prevent overfitting in neural networks. It works by randomly setting a fraction of the input units (neurons) to 0 at each update during training time. This means that these units are "dropped out" temporarily from the network, and their activations are ignored during that particular update.
![1_QrzcQNS2GS7J8wij0H8ANw](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/assets/112707550/76e7453e-fd54-421b-9828-727bdbc6a94f)
By randomly dropping out units, dropout prevents neurons from relying too much on the presence of any one input feature. It forces the network to learn more robust representations by preventing co-adaptation of neurons. In other words, dropout helps to ensure that no single neuron becomes too specialized and overly dependent on a particular feature or input. This reduces the chances of overfitting, as the network is forced to distribute its learning across a wider range of features.

### Project Overview
the sonar signal dataset is being used to discriminate between signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock. By applying dropout during training, the network is encouraged to learn more generalized features and avoid overfitting to specific training examples. This improves the model's ability to generalize and make accurate predictions on unseen data.

<b> For  detailed Implementation, Please Visit the [8_Dropout Regularization on sonar dataset.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/8_Dropout%20Regularization.ipynb)</b>

## File:-9_balancing ,unbalance dataset.ipynb

Imbalanced datasets are common in various domains such as customer churn rate, device failure prediction, and cancer prediction. The following methods can be used to address the class imbalance problem:

#### 1. Under-sampling the Majority Class

One approach is to randomly select a subset of samples from the majority class to balance the dataset. For example, you can randomly pick 1000 samples from the majority class (green) when you have 99000 samples and combine them with the minority class (red) samples to train the model.

#### 2. Over-sampling the Minority Class by Duplication

Another technique involves duplicating samples from the minority class to increase its representation. For instance, you can duplicate 1000 samples from the minority class (red) 99 times to match the number of samples in the majority class (green) and then train the model.

#### 3. Over-sampling the Minority Class using SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is a popular method for generating synthetic examples from the minority class. It creates synthetic samples by applying the k-nearest neighbor algorithm. This technique helps in increasing the diversity of the minority class.

#### 4. Ensemble Method

Ensemble methods can also be effective in handling imbalanced datasets. For example, if you have 3000 samples in the majority class (green) and 1000 samples in the minority class (red), you can create batches of 1000 green samples combined with 1000 red samples. Then, you can take the majority vote of the three batches to make predictions.

#### 5. Focal Loss

Focal Loss is a modification to the standard cross-entropy loss that helps address the class imbalance problem. It assigns more weight to the minority class samples and penalizes the majority class samples during the loss calculation. This approach can be particularly useful when there is a severe class imbalance.

<b> <b> For  detailed Implementation, Please Visit the [9_balancing ,unbalance dataset.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/9_balancing%20%2Cunbalance%20dataset.ipynb) </b>

## File:-10_convolutional neural network.ipynb

#### What is Convolutional Network(CNN)
A Convolutional Neural Network (CNN) is a type of deep learning algorithm designed for analyzing visual data, particularly images.

#### Introduction

In image classification, CNNs excel at learning representations directly from raw pixel data. They leverage the concept of convolutional layers, which utilize filters or feature detectors to identify important patterns or features within an image. The process involves convolving these filters over the input image to produce feature maps, which are then passed through subsequent layers for further processing and classification.

#### Feature Extraction and Classification

The CNN architecture consists of two main parts: feature extraction and classification.

#### Feature Extraction

In the feature extraction stage, the CNN applies various filters or feature detectors to the input image to capture different types of visual patterns. These filters act as feature detectors and identify specific characteristics, such as edges, textures, or shapes. Each filter is convolved with the input image, and the resulting feature maps highlight the presence of relevant features.

#### Classification

Once the features are extracted, the CNN flattens the feature maps and passes them through fully connected layers for classification. Additional layers, such as ReLU (Rectified Linear Unit) activation and pooling layers, are often utilized to introduce non-linearity and reduce the spatial dimensions of the feature maps, respectively. Finally, the output is passed through a softmax layer to obtain class probabilities.

### Benefits of Convolutional Neural Networks

#### Sparse Connectivity and Parameter Sharing

Unlike fully connected networks, CNNs leverage sparse connectivity, meaning that each neuron is only connected to a small region of the input data. This property reduces the number of parameters and enables the network to learn more efficiently. Additionally, CNNs utilize parameter sharing, where the same filter is applied across different regions of the input image. This sharing of parameters helps in detecting features regardless of their location within the image.


<b> For more Information, Please Visit the [10_convolutional neural network.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/10_convolutional%20neural%20network.ipynb) </b>

## File:-11_Image_classification_using_convolutional_neural_network.ipynb
## Project-This Project aims to perform image classification on the CIFAR-10 dataset using a convolutional Neural Network (CNN) Implemented in TensorFlow.
### Dataset Description
The CIFAR-10 dataset consist of 60000 32*32 color images in 10 different classes. It is divided into a training set of 50,000 images and the test set of 10,000 images. The task is to classify each image into one of the following categories✈️ airplane,automobile,bird,cat,deer,dog,frog,horse,ship or truck.
#### Model Architecture
1. Convolutional layer with 32 filters and a kernel size of (3, 3), using ReLU activation.
2. MaxPooling layer with a pool size of (2, 2).
3. Convolutional layer with 64 filters and a kernel size of (3, 3), using ReLU activation.
4. MaxPooling layer with a pool size of (2, 2).
5. Flatten layer to convert the 2D feature maps into a 1D feature vector.
6. Dense layer with 64 units and ReLU activation.
7. Dense layer with 10 units (corresponding to the 10 classes) and softmax activation for classification.

<b> For Detailed Implementation, Please Visit the [11_Image_classification_using_convolutional_neural_network.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/11_Image_classification_using_convolutional_neural_network.ipynb) </b>

## File:-12_flower classification using data_augmentation.ipynb
## Project
#### What is Data Augmentation
Data augmentation is a technique commonly used in machine learning and computer vision to artificially increase the size of a dataset by applying various transformations to the original data. The idea is to generate new examples that are similar to the original data, but with variations that make the model more robust to different scenarios and improve its generalization performance.

Some common data augmentation techniques include:

- Flipping and rotating images
- Cropping and resizing images
- Adding noise or distortion to images
- Changing the brightness, contrast, or color of images
  
Randomly applying combinations of these transformations For example, in image classification tasks, data augmentation can be used to generate new training images by randomly flipping, rotating, or cropping the original images. This can help the model learn to recognize the same object from different angles and orientations.  

<b> For Detailed Implementation, Please Visit the [12_flower classification using data_augmentation.ipynb](https://github.com/RNVALA/Deep-Learning-with-tensorflow-2.0-keras-and-python/blob/master/12_flower%20classification%20using%20data_augmentation.ipynb) </b>
