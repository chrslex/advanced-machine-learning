import numpy as np
from util import pooling2d

class Pooling2D(object):
  def __init__(self, pool_shape, stride, padding = 0, pool_mode = 'max'):
    self.pool_shape = pool_shape
    self.stride = stride
    self.padding = padding
    self.pool_mode = pool_mode
    self.activation = lambda x: x
    self.activation_deriv = lambda x: 1
    self.input_shape = None
    self.output_shape = None

    self.updateWBO()

  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    if (self.input_shape != None):
      self.output_shape = (((self.input_shape[0] + 2*self.padding - self.pool_shape[0])) // self.stride + 1,
                           ((self.input_shape[1] + 2*self.padding - self.pool_shape[1])) // self.stride + 1,
                           (self.input_shape[-1]))

  def forward(self, feature_maps):
    assert self.input_shape == feature_maps.shape[1:]
    result = []
    for fmap in feature_maps:
      result.append(pooling2d(fmap, self.pool_shape, self.stride, self.padding, self.pool_mode))

    result = np.array(result)
    self.output_shape = result.shape
    return result