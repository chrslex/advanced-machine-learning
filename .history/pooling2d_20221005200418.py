import numpy as np
from util import pooling2d


class Pooling2D(object):
  def __init__(self, pool_shape, stride, padding = 0, pool_mode = 'max'):
    # FEATURE MAP INPUT SHAPE:
    #   0: width
    #   1: height
    #   2: channel
    self.pool_shape = pool_shape
    self.stride = stride
    self.padding = padding
    self.pool_mode = pool_mode
    self.activation = lambda x: x
    self.activation_deriv = lambda x: 1
    self.input_shape = None
    self.output_shape = None

    self.backward_delta = {
      'max': self.maximum_backward_delta,
      'avg': self.average_backward_delta
    }

    self.updateWBO()

  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    if (self.input_shape != None):
      self.output_shape = (((self.input_shape[0] + 2*self.padding - self.pool_shape[0])) // self.stride + 1,
                           ((self.input_shape[1] + 2*self.padding - self.pool_shape[1])) // self.stride + 1,
                           (self.input_shape[-1]))

  def getSaveData(self):
    data = {
      'name': 'Pooling2D',
      'input_shape' : self.input_shape,
      'pool_shape': self.pool_shape,
      'stride': self.stride,
      'padding': self.padding,
      'pool_mode': self.pool_mode
      }

    return data

  def loadData(self, data):
    pass

  def forward(self, feature_maps):
    assert self.input_shape == feature_maps.shape[1:]
    result = np.zeros((
        feature_maps.shape[0], # num_of_feature_maps
        ((feature_maps.shape[1] + self.padding - self.pool_shape[0]) // self.stride) + 1, # width
        ((feature_maps.shape[2] + self.padding - self.pool_shape[1]) // self.stride) + 1, # height
        feature_maps.shape[3] # channel
      ))
    for idx, fmap in enumerate(feature_maps):
      result[idx] = pooling2d(fmap, self.pool_shape, self.stride, self.padding, self.pool_mode)

    self.output_shape = result.shape
    return result

  def average_backward_delta(self, neuron_input, delta, current_element, dx):
    each_batch, each_row, each_col, each_channel = current_element
    
    temp_pool = neuron_input[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ]

    # average = delta divided by input shape (width and height)
    average_delta = delta[each_batch, each_row, each_col, each_channel] / temp_pool.shape[0] / temp_pool.shape[1]

    dx[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ] += np.ones((temp_pool.shape[0], temp_pool.shape[1])) * average_delta
    return dx

  def maximum_backward_delta(self, neuron_input, delta, current_element, dx):
    each_batch, each_row, each_col, each_channel = current_element

    temp_pool = neuron_input[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ]
    # Mask True if element in pool is the max of the pool, else False
    masking = (temp_pool == np.max(temp_pool))
    dx[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ] += masking * delta[each_batch, each_row, each_col, each_channel]

    return dx

  def calcPrevDelta(self, neuron_input, delta, debug=False):
    dx = np.zeros(neuron_input.shape)

    for each_batch in range(delta.shape[0]):
      for each_row in range(delta.shape[1]):
        for each_col in range(delta.shape[2]):
          for each_channel in range(delta.shape[3]):
            # store each range variable to a variable, passing it easier to backward delta function
            current_element = [each_batch, each_row, each_col, each_channel]
            if (debug):
              print("Current Element:\n    batch  :", each_batch)
              print("    row    :", each_row)
              print("    column :", each_col)
              print("    channel:", each_channel)
            dx = self.backward_delta[self.pool_mode](neuron_input, delta, current_element, dx)
            if (debug):
              print("\n\nDX in this element batch after backward delta", dx)
              print("=============================================")

    return dx

  def backprop(self, neuron_input, delta, lr=0.001, debug=False):
    # no weight to update, only pass the error to previous layer
    return np.zeros(()), np.zeros(())

  def updateWeight(self, deltaWeight, deltaBias, debug=False):
    # no weight to update, only pass the error to previous layer
    pass