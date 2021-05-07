from tensorflow.keras import layers, datasets, models
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow import keras
import scipy
import tensorflow as tf
import math
import numpy as np
from sklearn.utils import class_weight
import os

"""
This module generates a CNN + multi-layer perceptron model or a soley CNN model with a using a data set of MFCC's divided into train, test, and validation sets.
Additionally, it provides modules that check the accuracy of the models on the testing dataset.
"""

def conv_block(x, num, filters, activation='relu'):
  """
  Secondary function for build_model2 and build_model to generate convolution block
  Parameters
  ----------
  x: tf.layer
    input layer
  
  num: int
    number of layers (sets of conv2d, batch norm and activation)
  
  filters: int 
    The dimensionality of the output space
  
  activation: str
    activation function to use for the convolutional layer

  Returns
  ------- 
    A convolution block for the CNN applied to x
  """

  for i in range(num):
      x = layers.Conv2D(filters, kernel_size = 7, padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation(activation)(x)

  return layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

def fully_connected_block(x, units):
  """
  Secondary function for build_model2 and build_model to generate fully connected block
  Parameters
  ----------
  x: tf.layer
    Input layer
  
  units: int
    Positive integer, dimensionality of the output space.

  Returns
  ------- 
    A fully connected block/layer applied to x
  """
  
  x = layers.Dense(units, activation="relu", kernel_regularizer='l2')(x)

  return layers.Dropout(0.1)(x)

def build_model2(input_shape1, input_shape2, num_classes, output_bias=None):
  """
  Builds a Neural Network Model that will use MFCC data as inputs to differentiate audio clips between covid positive and covid negative cases.
  Parameters
  ----------
  input_shape1: ndarray
    size of each MFCC data point
  
  input_shape2: ndarray
    size of each label for each data point
  
  num_classes: int
    number of classes
  
  BATCH_SIZE: int
    the number of training examples utilized in one iteration
  
  output_bias: tf.keras.initializers.Constant
    if given, a bias initializer for the last dense layer

  Return
  ------
  A model using CNN and multi-layer perceptron
  """
  
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)

  #Create Input layers
  input = layers.Input(shape=input_shape1, name='spec')
  input2 = layers.Input(shape=input_shape2, name='f')

  #CNN scheme
  x = layers.Conv2D(16, kernel_size=5, padding='same', activation='relu')(input) # Input block
  x = layers.MaxPool2D(pool_size=2, strides=2)(x)
  x = conv_block(x, 3, 32) # Main Convolution Blocks of 32 filters (before 12)
  x  = conv_block(x, 2, 32) # 64 filters (before 8)

  #Flatten and ready CNN for multi-layer perceptron
  x = layers.Flatten()(x) # Classification Layers for Convolutional block
  # x = fully_connected_block(x, 256)
  # x = fully_connected_block(x, 32)

  # Sequential Part
  x2 = layers.Dense(64, activation='relu', kernel_regularizer='l1')(input2) # (256, 1)
  x2 = layers.Dropout(0.5)(x2)
  # x2 = layers.Dense(16, activation='relu', kernel_regularizer='l1')(x2)
  # x2 = layers.Dropout(0.5)(x2)
  
  #x = layers.Add()([x, x2]) # Merge

  # x = layers.Flatten()(x)
  # x2 = layers.Flatten()(x2)
  x = layers.Concatenate()([x, x2]) # Concatenate CNN and multi-layer perceptron

  outputs = keras.layers.Dense(num_classes-1, activation="sigmoid", name='out', bias_initializer=output_bias)(x) # 2 classes, COVID, No COVID
  
  return keras.models.Model(inputs=[input, input2], outputs=outputs)
  
def build_model(input_shape, num_classes, output_bias=None):
  """
  Builds a Neural Network Model that will use MFCC data as inputs to differentiate audio clips between covid positive and covid negative cases.
  Parameters
  ----------
  input_shape: ndarray
    size of input data points
  
  num_classes: int
    number of classes
  
  BATCH_SIZE: int
    the number of training examples utilized in one iteration
  
  output_bias: tf.keras.initializers.Constant
    if given, a bias initializer for the last dense layer

  Returns
  ------- 
    A model using CNN only
  """
  
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)

  input = layers.Input(shape=input_shape) # Batch size parameter may not be needed
  
  x = layers.Conv2D(16, kernel_size=5, padding='same')(input) # Input block
  x = layers.Activation("relu")(x)
  x = layers.MaxPool2D(pool_size=2, strides=2)(x)
  # convolution blocks
  x = conv_block(x, 3, 32) # 12
  x  = conv_block(x, 2, 64) # 8
  # Flattening results from convolution
  x = layers.Flatten()(x)
  #x = fully_connected_block(x, 256)
  #x = fully_connected_block(x, 32)

  outputs = keras.layers.Dense(num_classes-1, activation="sigmoid", bias_initializer=output_bias)(x) # 2 classes, COVID, No COVID
  
  return keras.models.Model(inputs=input, outputs=outputs)

def get_class_weights(y_train):

  """
  Used to get a ration between the positive and negative cases so that the model can penalize the misclassification of the 
  minority class more than the misclassification of the majority class.

  Parameters
  ----------
  y_train: ndarray
    Labels of the dataset used to get the ratio between positive and negative cases

  Returns
  -------
  class_weight_dic: dict
    Dictionary with labels as the labels in the dataset and the keys as the ratio for the label
  
  """

  class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.reshape(-1, )), y_train.reshape(-1,))
  
  class_weight_dic = {0:min(class_weights), 1:max(class_weights)}

  return class_weight_dic

def build_train(new_model, x_train, y_train, BATCH_SIZE, epochs, lr, check_loc, name, val_data=None, show=True):
  """
  Trains cnn +  multi-layer perceptron model using input data from MFCCs
  Parameters
  ----------
  new_model: tf.model
    cnn +  multi-layer perceptron model
    
  x_train: ndarray
    MFCC train dataset
    
  y_train: ndarray
    Label of data points in x_train
    
  BATCH_SIZE: int
    the number of training examples utilized in one iteration
    
  epochs: int
    number of passes
    
  lr: int
    learning rate
    
  check_loc: str
    Location where to store the model weights

  val_data: tuple
    tuple of validation data used to determine overfitting
    
  show: bool
    Used to show the model summary

  Returns
  -------
  history: 
    The final trained model version of "new_model"

  """
  if show:
    new_model.summary()
    keras.utils.plot_model(new_model, show_shapes=True)
    
  weights = get_class_weights(y_train)
    
  new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="BinaryCrossentropy", metrics=["accuracy", 'binary_accuracy'])
    
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='auto', restore_best_weights=True)

  check_point = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(check_loc, name), save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

  history = new_model.fit(x=x_train, y=y_train, validation_data = val_data, batch_size=BATCH_SIZE, epochs=epochs, class_weight=weights, callbacks=[early_stopping, check_point])

  return history

def build_cnn_train(cnn_model, x_train, y_train, BATCH_SIZE, epochs, lr, check_loc, name, show=True, *args):
  """
  Trains cnn model using input data from MFCCs
  Parameters
  ----------
  cnn_model: tf.model
    Model using just CNN
    
  x_train: ndarray
    MFCC train dataset
    
  y_train: ndarray
    Label of data points in x_train
    
  BATCH_SIZE: int 
    the number of training examples in each partition utilized in one iteration
    
  epochs: int
    number of passes
    
  lr: int
    learning rate
    
  check_loc: str
    Location where to store the model weights
    
  show: bool
    Used to show the model summary

  args: tuple
    tuple of validation data used to determine overfitting

  Returns
  -------
  history: 
    The final trained model of "cnn_model"  using just CNN

  """
    
  if show:
    cnn_model.summary()
    keras.utils.plot_model(cnn_model, show_shapes=True)

  weights = get_class_weights(y_train)

  cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="BinaryCrossentropy",  metrics=["accuracy", 'binary_accuracy'])
  
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='auto', restore_best_weights=True)
  
  check_point = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(check_loc, name), save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

  history = cnn_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=args, class_weight=weights, callbacks=[early_stopping, check_point]) # Coughnet - 90, New - 81, Coswara
  
  return history