import numpy as np
from activation import *


class Dense:
    # Representasi 1 layer dalam ANN yang berisi beberapa perceptron
    def __init__(self, units, input_shape=None, activation=None):
        self.units = units
        # Input shape menyatakan bentuk input yang masuk ke layer ini
        self.input_shape = input_shape
        # Features menyatakan jumlah fitur dari input (akan dihitung dari input_shape)
        self.features = None
        self.activation_name = 'sigmoid' if activation == None else activation
        if(activation == None or activation == 'sigmoid'):
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif(activation == 'relu'):
            self.activation = relu
            self.activation_deriv = relu_deriv

        self.updateWBO()

    def updateInputShape(self, input_shape):
        self.input_shape = input_shape
        self.updateWBO()

    def updateWBO(self):
        # Update Weight, Bias, Output shape agar sesuai dengan units dan input_shape
        if(self.input_shape != None):
            # Handle kasus input shape berbentuk n-dimensi, ratakan menjadi (data_size, feature)
            # data_size = banyak data yang akan masuk nantinya (bervariasi)
            # feature = hasil kali seluruh fitur dari data, misal 1 data berbentuk array 2d (3, 4) maka feature = 3 * 4
            shape = list(self.input_shape)
            self.features = shape[0]
            for s in shape[1:]:
                self.features *= s
            # +1 untuk bias
            # Menggunakan Xavier Weight Initialization
            self.weight = np.random.randn(
                self.features+1, self.units) * np.sqrt(6/(self.features + self.units))
            # output_shape digunakan untuk menentukan input_shape layer berikutnya
            self.output_shape = (self.units,)

    def getSaveData(self):
        data = {'name': 'Dense',
                'unit': self.units, 
                'input_shape' : self.input_shape,
                'activation' : self.activation_name,
                'data' : {
                  'weight' : self.weight.tolist()
                }}
        return data
    
    def loadData(self, data):
      if('weight' not in data):
        raise KeyError("Data invalid")
      else:
        weight = np.array(data['weight'])
        if(weight.shape != self.weight.shape):
          raise TypeError("Weight shape invalid")
        else:
          self.set_weight(weight)

    def set_weight(self, weight):
        self.weight = weight.copy()

    def forward(self, x):
        # Melakukan feed-forward lalu mengembalikan output yang belum di aktifasi
        # Todo : Reshape x sesuai input_shape
        x = np.reshape(x, (-1, self.features))
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        v = np.reshape(np.dot(x, self.weight), tuple(
            [-1] + list(self.output_shape)))
        return v

    def calcPrevDelta(self, neuron_input, delta, debug=False):
        '''
          Hitung delta untuk layer sebelum layer ini dengan input layer ini & delta layer ini
          Input layer ini = output layer sebelumnya
        '''
        if(debug):
            print('Calculate previous delta for:')
            print('neuron_input: ', neuron_input)
            print('delta: ', delta)
        # Neuron input adalah output layer sebelumnya yang akan masuk sebagai input layer ini
        neuron_input = np.reshape(neuron_input, (-1, self.features))
        neuron_input = np.hstack(
            (np.ones((neuron_input.shape[0], 1)), neuron_input))
        # Calculate delta untuk backprop layer sebelumnya
        prev_delta = []
        for i in range(self.input_shape[-1]):
            # Hitung delta untuk unit ke-i layer sebelumnya
            # Delta unit-i layer hPrev= f'(out(hPrev)) * sum(weight[i+1] * delta layer h)
            # sum dilakukan sejumlah unit di layer ini
            out = self.activation_deriv(neuron_input[0, i+1])
            if(debug):
                print('    ', out, ' ', neuron_input[0, i+1])
            sum = 0
            for f in range(self.units):
                if(debug):
                    print('    self.weight[', i+1, ', ',
                          f, '] : ', self.weight[i+1, f])
                    print('    delta[', f, '] : ', delta[f])
                sum += self.weight[i+1, f] * delta[f]
                if(debug):
                    print('    sum : ', sum)
            prev_delta.append(out * sum)
        prev_delta = np.array(prev_delta)
        if(debug):
            print('Prev Delta : ', prev_delta)
        return prev_delta

    def updateWeight(self, deltaWeight, debug=False):
        # Update weight layer ini dengan {deltaWeight}
        if(debug):
            print('Weight : ')
            print(self.weight)
            print('Delta Weight : ')
            print(deltaWeight)
        self.weight += deltaWeight
        if(debug):
            print('Weight Now : ')
            print(self.weight)

    def backprop(self, neuron_input, delta, lr=0.001, debug=False):
        if(debug):
            print('Calculate backprop for:')
            print('neuron_input: ', neuron_input)
            print('delta: ', delta)
            print('lr: ', lr)
            print('input_shape: ', self.input_shape)
            print('units: ', self.units)
            print('weight_shape: ', self.weight.shape)
        # Neuron input adalah output layer sebelumnya yang akan menjadi input layer ini
        neuron_input = np.reshape(neuron_input, (-1, self.features))
        neuron_input = np.hstack(
            (np.ones((neuron_input.shape[0], 1)), neuron_input))

        # Calculate delta weight
        deltaWeight = np.zeros(self.weight.shape)
        # Untuk setiap input layer ini, hitung delta weightnya
        for idx, batch in enumerate(neuron_input):
            # Update semua isi deltaWeight yang berukuran sama seperti weight
            for i in range(self.input_shape[-1] + 1):
                # Untuk setiap unit di layer ini
                # (fully connected neural network, setiap input layer ini terhubung dengan setiap unit di layer ini)
                for j in range(self.units):
                    if(debug):
                        print('    ', lr, ' * ', delta[j], ' * ', batch[i])
                    deltaWeight[i, j] = lr * delta[j] * batch[i]
                    if(debug):
                        print('    deltaWeight: ', deltaWeight)

        return deltaWeight
