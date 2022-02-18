# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class Network:
    def __init__(self):
        #List of layers
        self.layers = []
        #Function counting loss
        self.loss = None
        #The deriative of function counting the loss
        self.loss_prime = None

    def add(self, layer):
        #Adding new layer to network
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        #Functions counting the loss
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        #Function counting output values
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        #Network training function
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                #Counting error only for getting information about
                #the error of network
                err += self.loss(y_train[j], output)

                #Backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            #Printing information about the average error
            err /= samples
            print(f"Iteracja nr {i+1} / {epochs}, blad = {err}")