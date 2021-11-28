#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Simple Perceptron
# The code tries to predict the output, given 3 binary inputs.
# If the second(middle) input is 1, the output should be one. Otherwise, the output should be 0.

import numpy as np
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust weights
def sigmoid_derivative(x):
    return x * (1 - x)

def neuron(inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, weights))
        return output
#===================================#
# training input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
# training output dataset
training_outputs = np.array([[0,1,0,1]]).T


# start with random weights
weights = 2 * np.random.random((3,1)) - 1   # 3x1 matrix
#===================================#

def train_neural_network(training_inputs, training_outputs, weights):
    # Iterate 20,000 times to train the neural network
    for i in range(20000):
        our_outputs = neuron(training_inputs) #starting with random weights

        error = training_outputs - our_outputs

        # multiply error by the slope of the sigmoid at the values in our outputs
        adjustments = error * sigmoid_derivative(our_outputs)

        # update weights: Less confident weights are adjusted more through the nature of the function
        weights += np.dot(training_inputs.T, adjustments)
    return weights     #we get updated weights after training


#=================================================#
weights = train_neural_network(training_inputs, training_outputs, weights) # weights variable is updated
#=================================================#
#================= New Situation =================#
A = str(input("Input 1: "))
B = str(input("Input 2: "))
C = str(input("Input 3: "))
    
print("New situation: input data = ", A, B, C)
print("Output data: ")
print(neuron(np.array([A, B, C])))

