import numpy as np
import cv2
import os


# Utility functions

# mat = batch * W * H * C
# kernel = num_filter * W_Kernel * H_Kernel * C_Kernel
# Output = batch * _ * _ * num_filter
def conv2d_batch(mat, kernel, pad, stride):
  padded_mat = np.pad(mat, pad)
  padded_mat_batch, padded_mat_x, padded_mat_y, padded_mat_c = padded_mat.shape
  num_filter, kernel_x, kernel_y, kernel_c = kernel.shape
  output_shape = (padded_mat_batch, (padded_mat_x - kernel_x) // stride + 1, (padded_mat_y - kernel_y) // stride + 1, num_filter)
  output = np.zeros(output_shape)

  for b in range(padded_mat_batch):
    for _filter in range(num_filter):
      for i in range(output_shape[1]):
        start_x = i*stride
        end_x = start_x + kernel_x
        for j in range(output_shape[2]):
          start_y = j*stride
          end_y = start_y + kernel_y 
          for chan in range(padded_mat_c):
            # print(b, i, j, _filter)
            output[b, i, j, _filter] += np.tensordot(padded_mat[b, start_x:end_x, start_y:end_y, chan], kernel[_filter, :, :, min(chan, kernel_c-1)]) 
          output[b, i, j, _filter] /= padded_mat_c
  return output

# mat = batch * W * H * C
# kernel = batch * num_filter * W_Kernel * H_Kernel * C_Kernel
# Output = batch * _ * _ * num_filter
def conv2d_batch_kernel(mat, kernel, pad, stride):
  padded_mat = np.pad(mat, pad)
  padded_mat_batch, padded_mat_x, padded_mat_y, padded_mat_c = padded_mat.shape
  kernel_batch,num_filter, kernel_x, kernel_y, kernel_c = kernel.shape
  if(padded_mat_batch != kernel_batch):
    print('Padded mat shape:', padded_mat.shape)
    print('kernel shape:', kernel.shape)
    raise ValueError("padded_mat_batch " + str(padded_mat_batch) + " does not match with kernel_batch " + str(kernel_batch))
  output_shape = (padded_mat_batch, (padded_mat_x - kernel_x) // stride + 1, (padded_mat_y - kernel_y) // stride + 1, num_filter)
  output = np.zeros(output_shape)

  for b in range(padded_mat_batch):
    for _filter in range(num_filter):
      for i in range(output_shape[1]):
        start_x = i*stride
        end_x = start_x + kernel_x
        for j in range(output_shape[2]):
          start_y = j*stride
          end_y = start_y + kernel_y 
          for chan in range(padded_mat_c):
            output[b, i, j, _filter] += np.tensordot(padded_mat[b, start_x:end_x, start_y:end_y, chan], kernel[b, _filter, :, :, min(chan, kernel_c-1)]) 
          output[b, i, j, _filter] /= padded_mat_c
  return output

def get_pooling_region(x, pool_shape, stride, output_shape):
  for i in range(output_shape[0]):
    for j in range(output_shape[1]):
      new_region = x[(i * stride):(i * stride + pool_shape[0]), (j * stride):(j * stride + pool_shape[1])]
      yield new_region, i, j

# Pooling function for 2d matrices
def one_channel_pooling(x_data, pool_shape, stride, padding, pool_mode = 'max'):
  # do this for each channel
  x = np.pad(x_data, padding , mode='constant')

  output_shape = (((x.shape[0] - pool_shape[0]) // stride) + 1,
                  ((x.shape[1] - pool_shape[1]) // stride) + 1)

  pool_output = np.zeros(output_shape)

  pooling_output_mode = {
    'max': np.amax,
    'avg': np.mean
  }

  for region, row, col in get_pooling_region(x, pool_shape, stride, output_shape):
    pool_output[row, col] = pooling_output_mode[pool_mode](region, axis=(0, 1))

  return pool_output

def pooling2d(x_data, pool_shape, stride, padding, pool_mode = 'max'):
  # pooling can be done on however many channel there is
  if (len(x_data.shape) == 2):
    # data consist of only single channel
    return one_channel_pooling(x_data, pool_shape, stride, padding, pool_mode)
  elif (len(x_data.shape) == 3):
    # data consist of n channels
    data = np.moveaxis(x_data, 2, 0) # change channels last to channels first formats

    pooling_output = np.zeros((
      data.shape[0], # channel
      ((data.shape[1] + padding - pool_shape[0]) // stride) + 1, # width
      ((data.shape[2] + padding - pool_shape[1]) // stride) + 1 # height
    ))

    for idx, data_channel in enumerate(data):
      pooling_output[idx] = one_channel_pooling(data_channel, pool_shape, stride, padding, pool_mode)

    return np.moveaxis(pooling_output, 0, 2) # change channels first to channels last format