"""
Transfer Learning API
"""
import numpy as np
import tensorflow as tf
from keras.models import Model

models = ['resnet', 'denseNet', 'vgg']

def expand_dim(matrix):
  """
  Expand the dimension of the input by one
  
  Parameters
  ----------
  matrix: ndarray
    array whose z axis we want to expand

  Returns
  -------
    expanded array
  """

  two = np.dstack((matrix, matrix))
  return np.dstack((matrix, two))

def expand_dim_all(x_train, x_test, x_val):

  """
  Expand the dimensions of the inputs for the transfer learning model
  
  Parameters
  ----------
  x_train: ndarray
    train partition of dataset

  x_test: ndarray
    test partition of dataset

  x_val: ndarray
    validation partition of dataset

  Returns
  -------
  x_train_3d: ndarray
    3 channel ndarray

  x_test_3d: ndarray
    3 channel ndarray

  x_val_3d: ndarray
    3 channel ndarray

  """
  x_train_3d = x_train.copy()
  x_test_3d = x_test.copy()
  x_val_3d = x_val.copy()

  x_train_3d = np.array(list(map(lambda x: expand_dim(x), x_train_3d)))
  x_test_3d = np.array(list(map(lambda x: expand_dim(x), x_test_3d)))
  x_val_3d = np.array(list(map(lambda x: expand_dim(x), x_val_3d)))
  
  return x_train_3d, x_test_3d, x_val_3d

def make_transfer_model(input_shape, model_type):
  """
  Parameters
  ----------
  input_shape: tuple
    input shape for the model

  model_type: str
    name of model to use

  Returns
  -------
  tf.keras.application.model
  
  """

  if model_type == 'resnet':
    transfer_learning_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
  elif model_type == 'denseNet':
    transfer_learning_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
  else:
    transfer_learning_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

  last_layer = transfer_learning_model.layers[-1].output
  flatten = tf.keras.layers.Flatten()(last_layer)
  outputs = tf.keras.layers.Dense(1, activation="sigmoid")(flatten)
  model = Model(transfer_learning_model.input, outputs)

  model.trainable = True
  if model_type == 'resnet':
    for layer in model.layers:
      set_trainable = False
      if layer.name in ['conv5_block3_3_conv', 'conv5_block3_2_relu', 'conv5_block3_2_conv']:
          set_trainable = True
      if set_trainable:
          layer.trainable = True
      else:
          layer.trainable = False

  if model_type == 'denseNet':
    for layer in model.layers:
      set_trainable = False
      if layer.name in ['conv5_block16_0_bn', 'conv5_block16_0_relu', 'conv5_block16_1_conv', 'conv5_block16_1_bn', 'conv5_block16_1_relu', 'conv5_block16_2_conv',
                        'conv5_block15_0_bn', 'conv5_block15_0_relu', 'conv5_block15_1_conv', 'conv5_block15_1_bn', 'conv5_block15_1_relu', 'conv5_block15_2_conv',
                        'conv5_block14_0_bn', 'conv5_block14_0_relu', 'conv5_block14_1_conv', 'conv5_block14_1_bn', 'conv5_block14_1_relu', 'conv5_block14_2_conv',
                        'conv5_block13_0_bn', 'conv5_block13_0_relu', 'conv5_block13_1_conv', 'conv5_block13_1_bn', 'conv5_block13_1_relu', 'conv5_block13_2_conv',
                        'conv5_block12_0_bn', 'conv5_block12_0_relu', 'conv5_block12_1_conv', 'conv5_block12_1_bn', 'conv5_block12_1_relu', 'conv5_block12_2_conv',
                        'conv5_block11_0_bn', 'conv5_block11_0_relu', 'conv5_block11_1_conv', 'conv5_block11_1_bn', 'conv5_block11_1_relu', 'conv5_block11_2_conv']:
          set_trainable = True
      if set_trainable:
          layer.trainable = True
      else:
          layer.trainable = False   

  if model_type == 'vgg':
    for layer in model.layers:
      set_trainable = False
      if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3 ', 'block5_conv4']:
          set_trainable = True
      if set_trainable:
          layer.trainable = True
      else:
          layer.trainable = False

  return model
