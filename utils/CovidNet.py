import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
import librosa
import librosa.display
import IPython.display as ipd
import tensorflow as tf
%tensorflow_version 2.x

from tensorflow.keras import layers, datasets, models
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow import keras


def conv_block(x, num, filters, activation='relu'):

  for i in range(num):
      x = layers.Conv2D(filters, kernel_size = 3, padding="same")(x)
      x = layers.BatchNormalization()(x)
  x = layers.Activation(activation)(x)
  return layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
  
  
def fully_connected_block(x, units):

  x = layers.Dense(units, activation="relu")(x)
  return layers.Dropout(0.25)(x)
  
def build_model(input_shape, num_classes):
  
  input = layers.Input(shape=input_shape) # idk if I should include batch_size=BATCH_SIZE as a parameters
  
  x = layers.Conv2D(16, kernel_size=5, padding='same')(input) # Input block
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.MaxPool2D(pool_size=2, strides=2)(x)

  x = conv_block(x, 3, 32)
  x  = conv_block(x, 3, 64)
  x = conv_block(x, 3, 128) # Taking this block make trainable params from 7m to 13m ??? idk y tho

  x = layers.Flatten()(x)
  x = fully_connected_block(x, 128)
  x = fully_connected_block(x, 32)

  outputs = keras.layers.Dense(num_classes, activation="softmax")(x) # 2 classes, COVID, No COVID
  
  return keras.models.Model(inputs=input, outputs=outputs)  
  
  
  
  