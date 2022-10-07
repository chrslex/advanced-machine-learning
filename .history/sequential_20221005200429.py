import numpy as np
import json
from dense import *
from conv2d import *
from pooling2d import *
from flatten import *
from sklearn.metrics import classification_report


class Sequential:
  # Representasi suatu model dimana perhitungan dilakukan sekuensial (layer per layer)
  def __init__(self):
    self.layers=[]
    self.output_shape = None
    self.lastDeltaWeight = None
    self.lastDeltaBias = None

  def add(self, layer):
    # Tambah layer baru ke model
    if(len(self.layers) == 0 and layer.input_shape == None):
      raise ValueError("Input shape cannot be null")
    else:
      if(self.output_shape == None):
        self.output_shape = layer.output_shape
      else:
        layer.updateInputShape(self.output_shape)
        self.output_shape = layer.output_shape
      self.layers.append(layer)

  def forward(self, x):
    # Lakukan feed-forward
    temp = x.copy()
    for l in self.layers:
      temp = l.activation(l.forward(temp))
    return temp

  def calculateError(self, yTrue, yPred):
    # MSE
    # return 0.5 * np.mean(np.square(yTrue.reshape(-1, 1) - yPred))

    # Binary Cross Entropy
    y = yTrue.reshape(-1, 1)
    return -1 * y * np.log(yPred) - (1 - y) * np.log(1 - yPred)

  def calculateLossDeriv(self, yTrue, yPred):
    # MSE
    # return yTrue.reshape(-1, 1) - yPred

    # Binary Cross Entropy
    y = yTrue.reshape(-1, 1)
    return (-y / yPred) + ((1 - y) / (1 - yPred))

  def saveModel(self, path):
    data = {'layers' : []}
    for layer in self.layers:
      data['layers'].append(layer.getSaveData())
    with open(path, 'w') as outfile:
      json.dump(data, outfile)

  def loadModel(self, path):
    with open(path) as json_file:
      data = json.load(json_file)
      for layer in data['layers']:
        if(layer['name'] == 'Dense'):
          self.add(Dense(layer['unit'], input_shape=layer['input_shape'], activation=layer['activation']))
          self.layers[-1].loadData(layer['data'])
        elif(layer['name'] == 'Conv2D'):
          self.add(Conv2D(layer['num_filter'], tuple(layer['kernel_shape']), pad=layer['pad'], stride=layer['stride'], input_shape=tuple(layer['input_shape']), activation=layer['activation']))
          self.layers[-1].loadData(layer['data'])
        elif(layer['name'] == 'Pooling2D'):
          self.add(Pooling2D(tuple(layer['pool_shape']), stride=layer['stride'], padding=layer['padding'], pool_mode=layer['pool_mode']))
        elif(layer['name'] == 'Flatten'):
          self.add(Flatten())
        else:
          raise TypeError("Unknown layer")

  def calcDelta(self, rawInput, yPred, yTarget, debug=False):
    # Menghitung delta output layer
    lastLayer = self.layers[-1]
    lastX = rawInput[-1]

    delta = lastLayer.activation_deriv(lastX) * self.calculateLossDeriv(yTarget, yPred)

    listDelta = []

    # Hitung delta setiap layer menggunakan layer setelahnya, mulai dari layer terakhir
    for i in range(len(self.layers)-1, -1, -1):
      listDelta.append(delta)
      delta = self.layers[i].calcPrevDelta(rawInput[i], delta, debug=debug)

    return listDelta

  # Fit n epoch, n batch
  def fit(self, xData, yData, lr=0.001, momentum=0, epochs=1, lr_decay=0, batch_size=1, debug=False):
    listErr = []
    lrNow = lr
    for e in range(epochs):
      print('Learning rate:', lrNow)
      self._fit_1_epoch(xData, yData, lr=lrNow, momentum=momentum, debug=debug, batch_size=batch_size)
      lrNow *= (1. / (1. + lr_decay * e))

      # Hitung error untuk epoch ini
      yPred = self.forward(xData)
      print(classification_report(yData, np.round(yPred)))
      epochErr = np.mean(self.calculateError(yData, yPred))

      # Simpan error dalam list agar bisa di plot nantinya
      listErr.append(epochErr)

      print('Epoch:', e+1, '= error : ', epochErr)
      print('======================================================\n\n')
    return listErr

  # Fit 1 epoch, n batch
  def _fit_1_epoch(self, xData, yData, lr=0.001, momentum=0, batch_size=1, debug=False):
    # Training dilakukan dengan konsep mini_batch, update weight dilakukan setiap {batch_size}
    numBatch = int(np.ceil(xData.shape[0] / batch_size))

    # Untuk setiap data dalam minibatch, hitung deltaWeightnya
    deltaWeight = []
    for iter in range(numBatch):
      start = iter * batch_size
      end = start + batch_size

      x = xData[start:end]
      y = yData[start:end]

      deltaWeight.append(self._fit_1_batch(x, y, lr=lr, debug=debug))

      # End for mini batch

    # Hitung total
    totalDeltaWeight = [] # List of delta weight tiap layer
    totalDeltaBias = [] # List of delta bias tiap layer
    for idx_layer in range(len(self.layers)):
      dw, db = deltaWeight[0][-1-idx_layer]
      for idx_batch in range(1, numBatch):
        dw += deltaWeight[idx_batch][-1-idx_layer][0]
        db += deltaWeight[idx_batch][-1-idx_layer][1]
      dw /= (numBatch * batch_size)
      db /= (numBatch * batch_size)
      totalDeltaWeight.append(dw)
      totalDeltaBias.append(db)

    if(self.lastDeltaWeight == None):
      self.lastDeltaWeight = totalDeltaWeight
    else:
      for idx in range(len(totalDeltaWeight)):
        totalDeltaWeight[idx] = momentum * self.lastDeltaWeight[idx] + (1-momentum) * totalDeltaWeight[idx]
      self.lastDeltaWeight = totalDeltaWeight

    if(self.lastDeltaBias == None):
      self.lastDeltaBias = totalDeltaBias
    else:
      for idx in range(len(totalDeltaBias)):
        totalDeltaBias[idx] = momentum * self.lastDeltaBias[idx] + (1-momentum) * totalDeltaBias[idx]
      self.lastDeltaBias = totalDeltaBias

    # Update weight
    print("Update weight")
    for idx, layer in enumerate(self.layers):
      print(layer, totalDeltaWeight[idx], totalDeltaBias[idx])
      layer.updateWeight(totalDeltaWeight[idx], totalDeltaBias[idx])
    print("-------------")

  # Fit 1 batch
  def _fit_1_batch(self, xData, yData, lr=0.001, debug=False):
    # rawInput berisi input setiap layer secara raw(belum di aktivasi layer sebelumnya)
    rawInput = [xData]

    # Lakukan feed-forward untuk data {data}
    temp = xData.copy()
    for l in self.layers:
      out = l.forward(temp)
      temp = l.activation(out)
      rawInput.append(out.copy())

    # Hitung delta setiap layer
    listDelta = self.calcDelta(rawInput, temp, yData, debug=debug)

    # Lakukan backpropagation untuk setiap layer, mulai dari layer terakhir
    # print('Backprop untuk setiap layer')
    deltaWeight = []
    for i in range(len(self.layers)-1, -1, -1):
      l = self.layers[i]
      dw, db = l.backprop(rawInput[i], listDelta[-1-i], lr, debug=debug)
      deltaWeight.append((
          np.expand_dims(np.sum(dw, axis=0), 0),
          np.expand_dims(np.sum(db, axis=0), 0)
      ))

    # Urutan deltaWeight = deltaWeight last layer, deltaWeight last layer - 1, ... deltaWeight first layer
    # print('Didalam fit_1_batch', deltaWeight)
    return deltaWeight