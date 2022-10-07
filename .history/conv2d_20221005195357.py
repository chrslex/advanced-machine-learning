import numpy as np
from activation import *
from util import *


class Conv2D:
  # kernel_shape = W * H
  def __init__(self, num_filter, kernel_shape, pad, stride, input_shape=None, activation='leaky_relu',):
    if (len(kernel_shape) != 2):
      raise ValueError("Kernel shape must be in 2 Dimension")
    self.num_filter = num_filter
    self.kernel_shape = kernel_shape
    self.kernel = None
    self.bias = np.zeros((num_filter,))
    self.pad_size = pad
    self.pad = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    self.stride = stride
    self.input_shape = input_shape
    self.output_shape = None
    self.activation_name = activation

    if (activation == 'relu'):
      self.activation = relu # detector part
      self.activation_deriv = relu_deriv
    elif (activation == 'sigmoid'):
      self.activation = sigmoid
      self.activation_deriv = sigmoid_deriv
    elif (activation == 'leaky_relu'):
      self.activation = leaky_relu
      self.activation_deriv = leaky_relu_deriv
    else:
      raise ValueError("Activation function " + activation + " does not exist.")

    if(input_shape != None):
      self.updateWBO()

  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    # Kernel = Num Filter * W * H * C_Input
    self.kernel = np.random.random((self.num_filter, self.kernel_shape[0], self.kernel_shape[1], self.input_shape[2]))

    self.output_shape = (((self.pad[1][0] + self.pad[1][1] + self.input_shape[0] - self.kernel.shape[1]) // self.stride) + 1,
                         ((self.pad[2][0] + self.pad[2][1] + self.input_shape[1] - self.kernel.shape[2]) // self.stride) + 1,
                         self.num_filter)
    
    self.kernel = self.kernel * np.sqrt(6/(np.sum(self.kernel.shape) + np.sum(self.output_shape)))

  def getSaveData(self):
    data = {
      'name': 'Conv2D',
      'num_filter': self.num_filter,
      'kernel_shape': self.kernel_shape,
      'pad': self.pad_size,
      'stride': self.stride,
      'input_shape': self.input_shape,
      'activation': self.activation_name,
      'data': {
        'kernel': self.kernel.tolist(),
        'bias': self.bias.tolist()
      }
    }
    return data

  def loadData(self, data):
    self.kernel = np.array(data['kernel'].copy())
    self.bias = np.array(data['bias'].copy())

  def forward(self, x):
    assert self.input_shape == x.shape[1:]
    return conv2d_batch(x, self.kernel, self.pad, self.stride)

  def calcPrevDelta(self, neuron_input, delta, debug=False):
    # Axis 0 = num filter, 1 = w, 2 = h, 3 = channel
    rotatedKernel = np.rot90(self.kernel, 2, (1, 2))  # rotasi 90 derajat 2 kali pada axis (1, 2)
    prevDelta = []
    activatedDerivInput = self.activation_deriv(neuron_input) # f'(z)

    deltaBatch, deltaW, deltaH, deltaC = delta.shape
    inpBatch, inpW, inpH, inpC = activatedDerivInput.shape
    rotKernelNum, rotKernelW, rotKernelH, rotKernelC = rotatedKernel.shape
    
    padW = (inpW - deltaW + rotKernelW - 1.0) / 2
    padH = (inpH - deltaH + rotKernelH - 1.0) / 2
    # Cek apakah pad negatif, jika iya maka buang sebagian delta, karena tidak berkontribusi terhadap weight (hasil padding)
    if(padW < 0):
      padW = -1 * padW
      left, right = int(np.ceil(padW)), int(np.floor(padW))
      delta = delta[:,left:-right]
      padW = 0
    if(padH < 0):
      padH = -1 * padH
      up, down = int(np.ceil(padH)), int(np.floor(padH))
      delta = delta[:, :, up:-down]
      padH = 0
    
    pad = (
        (0, 0),
        (int(np.ceil(padW)), int(np.floor(padW))),
        (int(np.ceil(padH)), int(np.floor(padH))),
        (0, 0),
    )

    if(debug):
      print('rotKernel shape:', rotatedKernel.shape)
      print('delta shape:', delta.shape)
      print('pad:', pad)

    tmp = conv2d_batch(delta, rotatedKernel, pad, 1) # full konvolusi antara rot(kernel) dan delta next layer
    # tmp = batch, w, h, c
    tmp = np.repeat(np.expand_dims(np.sum(tmp, axis=3), axis=3), inpC, axis=3) / tmp.shape[3]
    if(debug):
      print('tmp shape:', tmp.shape) # w, h, c
      print('activatedDerivInput shape:', activatedDerivInput.shape) # batch, w, h, c
    
    # for idx, inp in enumerate(activatedDerivInput):
    #   prevDelta.append(tmp[idx] * inp)
    # prevDelta = np.sum(np.array(prevDelta), axis=0) / activatedDerivInput.shape[0]
    prevDelta = tmp * activatedDerivInput
    if (debug):
      print('prevDelta shape:', prevDelta.shape)
      print('Conv2D delta shape:', prevDelta.shape)
      # print('Conv2D delta:', prevDelta)
      print('==================================================')
    return prevDelta

  # delta = batch, W, H, C
  # neuron input = batch, W, H, C
  def backprop(self, neuron_input, delta, lr=0.001, debug=False):
    res = np.zeros((1, delta.shape[3], *self.kernel.shape[1:]))
    inpBatch, inpW, inpH, inpC = neuron_input.shape
    deltaBatch, deltaW, deltaH, deltaC = delta.shape
    delta2 = np.expand_dims(delta, 4).swapaxes(3, 1).swapaxes(3, 2)

    if(debug):
      print('delta shape:', delta.shape) # W, H, C
      print('neuron_input shape:', neuron_input.shape)

    stride = max(int(np.floor((inpW - deltaW + self.pad[1][0] + self.pad[1][1])/(self.kernel.shape[2] - 1.0))), 1)
    tmp = conv2d_batch_kernel(self.activation(neuron_input), delta2, self.pad, stride)
    res = np.repeat(np.expand_dims(tmp, 4).swapaxes(3, 1).swapaxes(3, 2), inpC, axis=4)
    res = lr * np.sum(res, axis=0)
    if (debug):
      print('Conv2D backprop shape:', res.shape)
      print('==================================================')
    return res, np.zeros(())

  def updateWeight(self, deltaWeight, deltaBias, debug=False):
    self.kernel -= deltaWeight
    # self.bias += deltaBias