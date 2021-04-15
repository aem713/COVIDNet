from tensorflow.keras import layers, datasets, models
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow import keras
import scipy
import tensorflow as tf
import math
import numpy as np

"""
This module generates a CNN + multi-layer perceptron model or a soley CNN model with a using a data set of MFCC's divided into train, test, and validation sets.
Additionally, it provides modules that check the accuracy of the models on the testing dataset.
"""


def conv_block(x, num, filters, activation='relu'):
  """
  Secondary function for build_model2 and build_model to generate convolution block
  Inputs:
  - x: input layer
  - num: number of layers (sets of conv2d, batch norm and activation)
  - filters: The dimensionality of the output space

  Outputs: A convolution block for the CNN applied to x
  """

  for i in range(num):
      x = layers.Conv2D(filters, kernel_size = 3, padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation(activation)(x)

  return layers.MaxPool2D(pool_size=(2, 2), strides=2)(x) #



def fully_connected_block(x, units):
  """
  Secondary function for build_model2 and build_model to generate fully connected block
  Inputs:
  - x: Input layer
  - units: Positive integer, dimensionality of the output space.

  Outputs: A fully connected block/layer applied to x
  """
  
  x = layers.Dense(units, activation="relu", kernel_regularizer='l2')(x)

  return layers.Dropout(0.1)(x)

def build_model2(input_shape1, input_shape2, num_classes, BATCH_SIZE):
  """
  Builds a Neural Network Model that will use MFCC data as inputs to differentiate audio clips between covid positive and covid negative cases.
  Inputs:
  - input_shape1: size of each MFCC data point
  - input_shape2: size of each feature vector
  - num_classes: number of classes
  - BATCH_SIZE: the number of training examples utilized in one iteration

  Output(s): A model using CNN and multi-layer perceptron
  """
  #Create Input layers
  input = layers.Input(shape=input_shape1, batch_size=BATCH_SIZE, name='spec')
  input2 = layers.Input(shape=input_shape2, batch_size=BATCH_SIZE, name='f')

  #CNN scheme
  x = layers.Conv2D(16, kernel_size=5, padding='same')(input) # Input block
  x = layers.Activation("relu")(x)
  x = layers.MaxPool2D(pool_size=2, strides=2)(x)
  x = conv_block(x, 3, 12) # Main Convolution Blocks
  x  = conv_block(x, 2, 8)

  #Flatten and ready CNN for multi-layer perceptron
  x = layers.Flatten()(x) # Classification Layers for Convolutional block
  x = fully_connected_block(x, 256)
  x = fully_connected_block(x, 32)

  # Sequential Part
  x2 = layers.Dense(1000, activation='relu', kernel_regularizer='l1')(input2)
  x2 = layers.Dropout(0.1)(x2)
  x2 = layers.Dense(512, activation='relu', kernel_regularizer='l1')(x2)
  x2 = layers.Dropout(0.1)(x2)
  x2 = layers.Dense(256, activation='relu', kernel_regularizer='l1')(x2)
  
  #x = layers.Add()([x, x2]) # Merge

  x = layers.Concatenate()([x, x2]) # Concatenate CNN and multi-layer perceptron
  x = layers.Dense(32, activation='relu', kernel_regularizer='l1')(x)

  outputs = keras.layers.Dense(num_classes-1, activation="sigmoid", name='out')(x) # 2 classes, COVID, No COVID
  
  return keras.models.Model(inputs=[input, input2], outputs=outputs)

def build_model(input_shape, num_classes, BATCH_SIZE):
  """
  Builds a Neural Network Model that will use MFCC data as inputs to differentiate audio clips between covid positive and covid negative cases.
  Inputs:
  - input_shape: size of input data points
  - num_classes: number of classes
  - BATCH_SIZE: the number of training examples utilized in one iteration

  Output(s): A model using CNN only
  """
  
  input = layers.Input(shape=input_shape, batch_size=BATCH_SIZE) # Batch size parameter may not be needed
  
  x = layers.Conv2D(16, kernel_size=5, padding='same')(input) # Input block
  x = layers.Activation("relu")(x)
  x = layers.MaxPool2D(pool_size=2, strides=2)(x)
  # convolution blocks
  x = conv_block(x, 3, 12)
  x  = conv_block(x, 2, 8)
  # Flattening results from convolution
  x = layers.Flatten()(x)
  x = fully_connected_block(x, 256)
  x = fully_connected_block(x, 32)

  outputs = keras.layers.Dense(num_classes-1, activation="sigmoid")(x) # 2 classes, COVID, No COVID
  
  return keras.models.Model(inputs=input, outputs=outputs)
  
def build_train(new_model, x_train, y_train, BATCH_SIZE, epochs, show=True):
    """
    Trains cnn +  multi-layer perceptron model using input data from MFCCs
    Inputs:
    - new_model: cnn +  multi-layer perceptron model
    - x_train: MFCC traindataset
    - y_train: Label of data points in x_train
    - BATCH_SIZE: the number of training examples utilized in one iteration
    - epochs: number of passes

    Output(s):
    - history: The final trained model version of "new_model"

    """
  if show:
    new_model.summary()
    keras.utils.plot_model(new_model, show_shapes=True)
  new_model.compile(optimizer="Adam", loss="BinaryCrossentropy", metrics=["accuracy", tf.keras.metrics.PrecisionAtRecall(recall=0.8), tf.keras.metrics.SensitivityAtSpecificity(0.8)])
  history = new_model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs) # , validation_data=([x_val, x_val2], y_val) , validation_data={'spec': x_val, 'f': x_val2, 'out': y_val}, validation_batch_size=90
  return history

def build_cnn_train(cnn_model, x_train, y_train, BATCH_SIZE, epochs, val_batch_size, show=True, *args):
  """
   Trains cnn model using input data from MFCCs
    Inputs:
    - cnn_model: Model using just CNN
    - x_train: MFCC traindataset
    - y_train: Label of data points in x_train
    - BATCH_SIZE: the number of training examples utilized in one iteration
    - epochs: number of passes

    Output(s):
    -history: The final trained model of "cnn_model"  using just CNN

  """
    
  if show:
    cnn_model.summary()
    keras.utils.plot_model(cnn_model, show_shapes=True)
  cnn_model.compile(optimizer="Adam", loss="BinaryCrossentropy",  metrics=["accuracy", tf.keras.metrics.PrecisionAtRecall(recall=0.8), tf.keras.metrics.SensitivityAtSpecificity(0.8)])
  history = cnn_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=args, validation_batch_size=val_batch_size) # Coughnet - 90, New - 81, Coswara
  return history

def get_accuracy(new_model, x_test, y_test):
  """
  Checks the accuracy of the model on the test set of data
  Inputs:
  - new_model: Trained model
  - x_test: Testing Dataset
  - y_test: Label of data points in x_test

  Outputs:
  - y_hat2: matrix showing which points were
  - MSE: Mean squared error
  - RMSE: Root mean squared error
  - Accuracy: Accuracy of the model to predict covid condition on testing data


  """

  y = y_test.copy()
  y = y.reshape(-1,)

  prediction = new_model(x_test, training=False)
  proto_tensor = tf.make_tensor_proto(prediction)
  y_hat = tf.make_ndarray(proto_tensor)
  y_hat = y_hat.reshape(-1,)
  y_hat2 = np.round(y_hat)

  MSE = np.square(np.subtract(y,y_hat2)).mean() 
  RMSE = math.sqrt(MSE)
  correct = y_hat2 == y
  Accuracy = sum(correct)/correct.shape[0]

  return y_hat2, MSE, RMSE, Accuracy
