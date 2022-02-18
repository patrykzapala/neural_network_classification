# -*- coding: utf-8 -*-

class ActivationLayer():
    def __init__(self, activation, d_activation):
        #Setting the activation function
        self.activation = activation
        self.d_activation = d_activation

    def forward_propagation(self, input_data):
        #Counting output value from layer
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        #Backward propagation
        return self.d_activation(self.input) * output_error