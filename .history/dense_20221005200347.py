import numpy as np
from activation import *


class Dense:
    # Representasi 1 layer dalam ANN yang berisi beberapa perceptron
    def __init__(self, units, time_distributed=False, weight_initializer='default', input_shape=None, activation='sigmoid'):
        self.units = units
        # Input shape menyatakan bentuk input yang masuk ke layer ini
        self.input_shape = input_shape
        # Features menyatakan jumlah fitur dari input (akan dihitung dari input_shape)
        self.features = None
        self.activation_name = activation
        self.weight_initializer = weight_initializer
        self.time_distributed = time_distributed

        if (activation == 'sigmoid'):
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif (activation == 'relu'):
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif (activation == 'leaky_relu'):
            self.activation = leaky_relu
            self.activation_deriv = leaky_relu_deriv
        elif (activation == 'softmax'):
            if self.time_distributed:
                self.activation = softmax_time_distributed
                self.activation_deriv = softmax_time_distributed_deriv
            else:
                self.activation = softmax
                self.activation_deriv = softmax_deriv
        else:
            raise ValueError("Activation function " + activation + " does not exist.")

        self.updateWBO()

    def updateInputShape(self, input_shape):
        self.input_shape = input_shape
        self.updateWBO()

    def initWeight(self, size):
        if self.weight_initializer == 'random':
            return np.random.random(size)
        elif self.weight_initializer == 'zeros':
            return np.zeros(size)
        elif self.weight_initializer == 'ones':
            return np.ones(size)

    def updateWBO(self):
        # Update Weight, Bias, Output shape agar sesuai dengan units dan input_shape
        if (self.input_shape != None):
            # Handle kasus input shape berbentuk n-dimensi, ratakan menjadi (data_size, feature)
            # data_size = banyak data yang akan masuk nantinya (bervariasi)
            # feature = hasil kali seluruh fitur dari data, misal 1 data berbentuk array 2d (3, 4) maka feature = 3 * 4
            shape = list(self.input_shape)
            if self.time_distributed:
                self.features = 1
            else:
                self.features = shape[0]

            for s in shape[1:]:
                self.features *= s
            # +1 untuk bias
            if self.weight_initializer == 'default':
                # Menggunakan Xavier Weight Initialization
                self.weight = np.random.randn(
                    self.features, self.units) * np.sqrt(6/(self.features + self.units))
                self.bias = np.zeros((1, self.units))
            else:
                self.weight = self.initWeight((self.features, self.units))
                self.bias = self.initWeight((1, self.units))
            # output_shape digunakan untuk menentukan input_shape layer berikutnya
            self.output_shape = (self.units,)

    def getSaveData(self):
        data = {'name': 'Dense',
                'unit': self.units, 
                'input_shape' : self.input_shape,
                'activation' : self.activation_name,
                'data' : {
                  'weight' : self.weight.tolist(),
                  'bias': self.bias.tolist()
                }}
        return data

    def loadData(self, data):
      if ('weight' not in data):
        raise KeyError("Data invalid")
      else:
        weight = np.array(data['weight'])
        self.bias = np.array(data['bias']).copy()
        if (weight.shape != self.weight.shape):
          raise TypeError("Weight shape invalid")
        else:
          self.set_weight(weight)

    def set_weight(self, weight):
        self.weight = weight.copy()

    def forward(self, x):
        if (self.time_distributed):
            x = np.reshape(x, (self.input_shape[0], self.features))
            result = np.dot(x[0:1], self.weight) + self.bias
            for i in range(1, self.input_shape[0]):
                temp = np.dot(x[i:i+1], self.weight) + self.bias
                result = np.vstack((result, temp))
            return result
        else:
            x = np.reshape(x, (-1, self.features))
            return np.dot(x, self.weight) + self.bias

    def calcPrevDelta(self, neuron_input, delta, debug=False):
        tmp = self.activation_deriv(neuron_input)
        return np.multiply(tmp, np.dot(delta, self.weight.T))

    def updateWeight(self, deltaWeight, deltaBias, debug=False):
        self.weight -= deltaWeight
        self.bias -= deltaBias

    def backprop(self, neuron_input, delta, lr=0.001, debug=False):
        return lr * np.dot(self.activation(neuron_input.T), delta), delta