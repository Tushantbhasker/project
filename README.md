# Handwritten Digit Recognition
In this project we are going to classify the handwritten digits between 0-9. We will be using the famous MNIST dataset
for tarining the model. Also we will be developing a web application to deploy our model. This application will be a 
flask based seb application.
For the classification model we will be developing a CNN model.

## Project Overview
* Prepare dataset
* Develop CNN Model
* Training
* Evaluate Model
* Web application

## Python Environment
To run this progarm you have a Python SciPy environment installed, ideally with Python 3. You must have Keras (2.1.5 
or higher) installed with either the TensorFlow or Theano backend. You should also have scikit-learn, Pandas, NumPy, and 
Matplotlib installed.

## Dataset
I have used MNIST dataset provided by Keras.<br/>

## Defining the Model
I have used a CNN model for training.
* First layer is a 2-D Convolutional layer with 32 nodes.
* Second layer is also a 2-D Convolutional layer but with a 64 nodes.
* Next is a Dense layer wih a 128 nodes and Relu as a activation fuction.
* Last layer is output layer with 10 nodes and Softmax as a activation function. 

## Output
![Inout Image](/output.png)<br/>
__Predicted Output: __ 8
## Dependencies
* Keras
* Tensorflow
* Numpy
* pandas
* Matplotlib
* Pickle
* OS module
* Flask
