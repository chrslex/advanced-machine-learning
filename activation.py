import numpy as np

# Activation Function & Turunannya
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
  return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
  return np.maximum(x, 0)

def relu_deriv(x):
  return np.heaviside(x, 0)
