# -*- coding: utf-8 -*-
 
import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        #Generating random weights and biases
        self.weights = np.random.rand(input_size, output_size) -0.5
        self.bias = np.random.rand(1, output_size) -0.5

    def forward_propagation(self, input_data):
    	#Counting output value from layer
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        #Backward propagation
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        #Updating weights and biases
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        
        return input_error