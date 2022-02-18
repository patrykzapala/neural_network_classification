# -*- coding: utf-8 -*-

import numpy as np

def mse(y_true, y_pred):
    #MSE
    return np.mean(np.power(y_true-y_pred, 2));

def d_mse(y_true, y_pred):
    #Deriative of MSE
    return 2*(y_pred-y_true)/y_true.size;

def tanh(x):
    #Hyperbolic tangent function
    return np.tanh(x);

def d_tanh(x):
    #Deriative of hyperboic tangent
    return 1-np.tanh(x)**2;

def sigmoid(x):
    #Sigmoid function
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
	#Deriative of sigmoid function
	return sigmoid(x) * (1 - sigmoid(x))

def linear(x):
    #Linear funtion
    return x

def d_linear(x):
    #Deriative of linear function
    return 1.0
