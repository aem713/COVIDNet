# -*- coding: utf-8 -*-
"""
SurfboardFeatures API
"""

import librosa
import numpy as np
import scipy
from sklearn.decomposition import PCA
import random
from tensorflow.keras import layers, datasets, models, Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow import keras
import scipy
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Input
from tqdm import tqdm
import sklearn
import surfboard
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import os

from utils import Parameters

SAMPLING_RATE = Parameters.SAMPLING_RATE
DURATION = Parameters.DURATION
NUM_MELS = Parameters.NUM_MELS
FMAX = Parameters.FMAX
TOP_DB = Parameters.TOP_DB
EPS = Parameters.EPS
N_MFCC = Parameters.N_MFCC
PCA_COMPONENTS = Parameters.PCA_COMPONENTS
components_list = Parameters.components_list
statistics_list = Parameters.statistics_list

def myfunction():
  """
  Returns the value 0.35 as an input to the random.shuffle() function
  inputs:
    - N/A
  outputs:
    - the number 0.35
  """
  return 0.35

def GenerateVectors(path, dataset_type):
  """
  Computes the normalized feature vectors of the input path
  inputs:
    - path: the path containing all the desired signals to process
    - dataset_type: the type of the dataset (train/test/val)
  outputs:
    - x_2: a numpy array containing all the feature vector of the signals contained
      in the input
  """

  positive_paths = []
  negative_paths = []

  #Iterates over positive files in the path and appends them to positive_paths
  for entry in os.scandir(os.path.join(path, os.path.join(dataset_type, 'pos'))):
    if entry.is_file():
      positive_paths.append(entry.path)
  
  #Iterates over negative files in the path and appends to negative_paths
  for entry in os.scandir(os.path.join(path, os.path.join(dataset_type, 'neg'))):
    if entry.is_file():
      negative_paths.append(entry.path)
        
  #Computes the normalized feature vectors of the positive waveforms
  waveforms_positive = []
  for paths in tqdm(positive_paths):
    waveforms_positive.append(Waveform(path=paths, sample_rate=SAMPLING_RATE))
  postive_features_df = extract_features(waveforms=waveforms_positive, components_list=components_list, statistics_list=statistics_list)
  positive_features = sklearn.preprocessing.normalize(postive_features_df.to_numpy(), axis=0)

  #Computes the normalized feature vectors of the negative signals
  waveforms_negative = []
  for npaths in tqdm(negative_paths):
    waveforms_negative.append(Waveform(path=npaths, sample_rate=SAMPLING_RATE))
  negative_features_df = extract_features(waveforms=waveforms_negative, components_list=components_list, statistics_list=statistics_list)
  negative_features = sklearn.preprocessing.normalize(negative_features_df.to_numpy(), axis=0)

  #Combines the two feature vector lists
  positive_features_list = positive_features.tolist()
  negative_features_list = negative_features.tolist()

  x_2_list = positive_features_list.copy()
  x_2_list.extend(negative_features_list)

  #Shuffle the list
  random.shuffle(x_2_list, myfunction)

  x_2 = np.array(x_2_list)
  
  return x_2  


def Get_feature_vectors(new_audio_path, coswara_audio_path, audio_path, loc, load=True):
  """
  Function to get the data vectors 
  Parameters
  ----------
  new_audio_path: str
    audio path of the new dataset
  
  coswara_audio_path:str
    audio path of the coswara dataset
  
  audio_path:str
    audio path to use

  load: bool
    Whether to load or generate vectors
      True: Load vectors
      False: Generate vectors: Warning takes around 1hr

  Returns
  -------
  test_data_vec: np array
    testing vectors for the model

  train_data_vec: np array
    training vectors for the model
  
  val_data_vec: np array
    validatoin vectors for the model
  """
  if load:
    if audio_path == new_audio_path:
      test_data_vec = np.load(os.path.join(loc, 'Test_Vector_NewDataset.npy')) # JUST LOAD THE VECTORS
      train_data_vec = np.load(os.path.join(loc, 'Train_Vector_NewDataset.npy'))
      val_data_vec = np.load(os.path.join(loc, 'Val_Vector_NewDataset.npy'))
    elif audio_path == coswara_audio_path:
      test_data_vec = np.load(os.path.join(loc, 'Test_Vector_Coswara.npy')) # JUST LOAD THE VECTORS
      train_data_vec = np.load(os.path.join(loc, 'Train_Vector_Coswara.npy'))
      val_data_vec = np.load(os.path.join(loc, 'Val_Vector_Coswara.npy'))
  
  else:
    test_data_vec = GenerateVectors(audio_path, 'test') #    90, 404, 515
    train_data_vec = GenerateVectors(audio_path, 'train') # 720, 1292, 1670
    val_data_vec = GenerateVectors(audio_path, 'val') #      90, 324, 422
    if audio_path == new_audio_path:
      np.save(os.path.join(loc, 'Test_Vector_NewDataset'), test_data_vec)
      np.save(os.path.join(loc, 'Train_Vector_NewDataset'), train_data_vec)
      np.save(os.path.join(loc, 'Val_Vector_NewDataset'), val_data_vec)
    elif audio_path == coswara_audio_path:
      np.save(os.path.join(loc, 'Test_Vector_Coswara'), test_data_vec)
      np.save(os.path.join(loc, 'Train_Vector_Coswara'), train_data_vec)
      np.save(os.path.join(loc, 'Val_Vector_Coswara'), val_data_vec)

  return test_data_vec, train_data_vec, val_data_vec

def Reduce_Dim(n_components, train, test, val):

  """
  Dimensionality reduction function

  Parameters
  ----------
  n_components: int
    Number of components to keep

  train: ndarray
    Original training array from Generate_Vectors

  test: ndarray
    Original testing array from Generate_Vectors

  val: ndarray
    Original validation array from Generate_Vectors

  Returns
  -------
  x_test2: ndarray
    Dimensionally reduced testing arrays

  x_train2: ndarray
    Dimensionally reduced training arrays

  x_val2: ndarray
    Dimensionally reduced validation arrays
  
  """
  assert (len(train.shape)==2), 'Train input should be a np array of size (N, M)'
  assert (len(test.shape)==2), 'Test input should be a np array of size (N, M)'
  assert (len(val.shape)==2), 'Validation input should be a np array of size (N, M)'

  assert (train.shape[0]>=n_components), 'Number of samples in train input must be larger than n_components'
  assert (test.shape[0]>=n_components), 'Number of samples in test input must be larger than n_components'
  assert (val.shape[0]>=n_components), 'Number of samples in val input must be larger than n_components'

  pca = PCA(n_components=n_components)
  x_test2 = pca.fit(train).transform(test)
  x_train2 = pca.fit(train).transform(train)
  x_val2 = pca.fit(train).transform(val)
  
  return x_test2, x_train2, x_val2