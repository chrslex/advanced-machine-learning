import numpy as np
import json
from dense import *
from flatten import *

class Sequential:
  # Representasi suatu model dimana perhitungan dilakukan sekuensial (layer per layer)
  def __init__(self):
    self.layers=[]
    self.output_shape = None

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

  def calculateError(self, x, y):
    # Menghitung Mean Squared Error dari y dan hasil feed forward x
    temp = self.forward(x)
    return 0.5 * np.mean(np.square(y - temp))

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
        elif(layer['name'] == 'Flatten'):
          self.add(Flatten())
          self.layers[-1].loadData(layer['data'])
        else:
          raise TypeError("Unknown layer")