#handwritten-digit-analyzer

#Loading the libraries for use
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#loading the data 
data = pd.read_csv('train (1).csv')
data = np.array(data)
m, n = data.shape

#Splitting the data into training data and test data
np.random.shuffle(data)
split = int(0.8 * m)
train_data = data[:split, :]
test_data  = data[split:, :]

X_train = train_data[:, 1:].T
X_train = X_train / 255.0
Y_train = train_data[:, 0]

X_test = test_data[:, 1:].T
X_test = X_test / 255.0
Y_test = test_data[:, 0]

#Random Initialisation of weights and biases and subtracting 0.5 to make the range from (-0.5 to 0.5)
def initialize_parameters():
  W1 = np.random.rand(10, 784) - 0.5
  B1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  B2 = np.random.rand(10, 1) - 0.5
  return W1, B1, W2, B2

#An activation function
def ReLU(X):
  return np.maximum(X, 0)

#Subtracting the max from vector by all values to bring the max value to 0
#Then applying np.exp and dividing all values by the total sum, resulting in a vector of percentages
def softmax_calculator(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

#Standard calulation through the network
def forward_propagation(W1, B1, W2, B2, X):
  Z1 = W1.dot(X) + B1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + B2
  A2 = softmax_calculator(Z2)
  return Z1, A1, Z2, A2

#Converting the labels into one hot encoded vectors
def one_hot_converter(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  return one_hot_Y.T

#Finding the partial derivatives of all the parameters
def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
  one_hot_Y = one_hot_converter(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  dB2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
  dW1 = 1 / m * dZ1.dot(X.T)
  dB1 = 1 / m * np.sum(dZ1)
  return dW1, dB1, dW2, dB2

#Updating the parameters(weights and biases) by the chosen learning rate
def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
  W1 = W1 - learning_rate * dW1
  B1 = B1 - learning_rate * dB1
  W2 = W2 - learning_rate * dW2
  B2 = B2 - learning_rate * dB2
  return W1, B1, W2, B2

#Takes the highest value's index as prediction
def get_predictions(A2):
  return np.argmax(A2, 0)

#Gets the accuracy of model by summing the total correct predictions and diving by size
def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

#Applying all the algorithms and displaying the accuracy as the model trains
def gradient_descent(X, Y, alpha, iterations):
  W1, B1, W2, B2 = initialize_parameters()

  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
    dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
    W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

    if (i%20)==0:
      print("Iteration number: ", i)
      print("Accuracy = ", get_accuracy(get_predictions(A2), Y))
  return W1, B1, W2, B2

#This line runs the whole model essentially
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.10, 1000)

#Testing the model with the label at index 464
index = 464
Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X_test)
  
current_image = X_test[:, index].reshape((28, 28)) * 255
print("Prediction: ", get_predictions(A2)[index])
print("Label: ", Y_test[index])
plt.gray()
plt.imshow(current_image, interpolation='nearest')
plt.show()
