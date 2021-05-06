import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import tensorflow as tf
import random
from utils import Parameters

SAMPLING_RATE = Parameters.SAMPLING_RATE
DURATION = Parameters.DURATION
NUM_MELS = Parameters.NUM_MELS
FMAX = Parameters.FMAX
TOP_DB = Parameters.TOP_DB
EPS = Parameters.EPS

def create_mel(path, sr = SAMPLING_RATE, n_mels=512, fmax = 10000, top_db=30, eps = 1e-6, plot = False):
  """
  Creates a Mel Spectrogram from a given audio sample. This code standardizes 
  important aspects such as sampling rate, duration, normalization, and other
  related aspects of the audio

  Parameters:
  ----------
  path: string
    The file location of the audio sample
  
  sr: int
    Sampling rate with which to sample the audio
  
  n_mels: int
    Length of the FFT window
  
  fmax: int
    The maximum frequency
   
  top_db: int
    The threshold (in decibels) below reference to consider as silence
   
  eps: int
    Variation
   
  plot: boolean
    A boolean which decides whether to create a plot or not

  Returns:
  --------
  A scaled and normalized mel spectrogram array

  """
  # Loads the path and trims the top_db
  y , sr = librosa.load(path, sr=sr)
  yt, index = librosa.effects.trim(y, top_db=top_db)

  # Standardizes the length
  if yt.shape[0] < DURATION*SAMPLING_RATE:
    yt=np.pad(yt,int(np.ceil((DURATION*sr-yt.shape[0])/2)), mode='reflect')
  else:
    yt=yt[:DURATION*sr]

  # S = librosa.feature.mfcc(y=yt, sr=SAMPLING_RATE,n_mfcc=13)
  
  # mean = S.mean()
  # std = S.std()
  # S_norm = (S - mean) / (std + eps)

  # Plotting code if plot is True
  if plot:
    plt.figure()
    librosa.display.specshow(S)
    plt.colorbar()


  if plot:
    plt.figure()
    librosa.display.specshow(S_norm)
    plt.colorbar()

  # Produces the Mel Spectrogram 
  S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=n_mels, fmax=fmax)
  S_dB = librosa.power_to_db(S, ref=np.max)
  mean = S_dB.mean()
  std = S_dB.std()
  S_dB_norm = (S_dB - mean) / (std + eps)
  
  # Normalization of the Mel Spectrogram
  S_dB_min, S_dB_max = S_dB_norm.min(), S_dB_norm.max()
  S_dB_scaled = 255 * (S_dB_norm - S_dB_min) / (S_dB_max + eps - S_dB_min)
  S_dB_scaled = S_dB_scaled.astype(np.uint8)
  
  #if plot:
  #  plt.figure()
  #  librosa.display.specshow(S_dB_scaled)
  #  plt.colorbar()

  return S_dB_scaled.reshape(S_dB_scaled.shape[0], S_dB_scaled.shape[1], 1)

def myfunction():
  """
  Returns the number 0.35. This is used in the random.shuffle() function

  Parameters:
  -----------
    N/A
  
  Returns:
  --------
    The number 0.35

  """
  return 0.35

def DataGenerator(sr, n_mels, fmax, top_db, eps, path, dataset_type='train'):
  """
  Creates Mel Spectrograms for all the files in the path
  
  Parameters:
  -----------
  path: string
    The file location of the audio sample
  
  sr: int
    Sampling rate with which to sample the audio
  
  n_mels: int
    Length of the FFT window
   
  fmax: int
    The maximum frequency
  
  seed: float
    The seed with which to randomly shuffle
  
  top_db: int
    The threshold (in decibels) below reference to consider as silence
  
  eps: int
    Variation
   
  dataset_type: string
    The type of data set such as train/test/val. 
  
  Returns:
  --------
    x_1: numpy array
      The collection of mel spectrograms of positive and negative samples
    
    y_1: numpy array
      A list of 1's and 0's representing positive and negative samples

  """
  # Initialize the paths and append the paths to the list
  positive_paths = []
  negative_paths = []
  for entry in os.scandir(os.path.join(path, os.path.join(dataset_type, 'pos'))):
    if entry.is_file():
      positive_paths.append(entry.path)
        
  for entry in os.scandir(os.path.join(path, os.path.join(dataset_type, 'neg'))):
    if entry.is_file():
      negative_paths.append(entry.path)

  # Create the Mel Spectrogram for all the files in the path list as well as create labels of infection status
  path_to_audio_p = list(map(lambda x: create_mel(x, SAMPLING_RATE, NUM_MELS, FMAX, TOP_DB, EPS, plot=False), positive_paths))
  pos_labels_ds = [1 for i in positive_paths]

  path_to_audio_n = list(map(lambda x: create_mel(x, SAMPLING_RATE, NUM_MELS, FMAX, TOP_DB, EPS, plot=False), negative_paths))
  neg_labels_ds = [0 for i in negative_paths]

  # Organize all the paths and labels together
  x_list = path_to_audio_p.copy()
  y_list = pos_labels_ds.copy()

  x_list.extend(path_to_audio_n)
  y_list.extend(neg_labels_ds)

  # Randomly shuffle the paths and the labels
  random.shuffle(x_list, myfunction)
  random.shuffle(y_list, myfunction)

  # Return a numpy array of the list of the paths and the labels
  x_1 = np.array(x_list)
  y_1 = np.array(y_list)

  return x_1, y_1


def Get_matrix_inputs(new_audio_path, coswara_audio_path, audio_path, loc, load=True):
  """
  Function to get the spectrogram images 

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
  x_test: np array
    testing images for the model

  y_train: np array
    labels for testing dataset

  x_train: np array
    training images for the model

  y_train: np array
    labels for testing dataset
  
  x_val: np array
    validatoin images for the model

  y_val: np array
    labels for validation dataset

  """
  # If we want to simply load the vectors, we can do so by importing them from the Google Drive
  if load:

    # If we want to load the data from the coswara_audio_path
    if audio_path == coswara_audio_path:
      x_train = np.load(os.path.join(loc, 'X_Train_Spec_Coswara_15.npy')) # JUST LOAD THE VECTORS
      y_train = np.load(os.path.join(loc, 'Y_Train_Spec_Coswara_15.npy'))
      x_test = np.load(os.path.join(loc, 'X_Test_Spec_Coswara_15.npy'))
      y_test = np.load(os.path.join(loc, 'Y_Test_Spec_Coswara_15.npy'))
      x_val = np.load(os.path.join(loc, 'X_Val_Spec_Coswara_15.npy'))
      y_val = np.load(os.path.join(loc, 'Y_Val_Spec_Coswara_15.npy'))
 
    # If we want to load the data from the new_audio_path
    elif audio_path == new_audio_path:
      x_train = np.load(os.path.join(loc, 'X_Train_Spec_NewDataset.npy')) # JUST LOAD THE VECTORS
      y_train = np.load(os.path.join(loc, 'Y_Train_Spec_NewDataset.npy'))
      x_test = np.load(os.path.join(loc, 'X_Test_Spec_NewDataset.npy'))
      y_test = np.load(os.path.join(loc, 'Y_Test_Spec_NewDataset.npy'))
      x_val = np.load(os.path.join(loc, 'X_Val_Spec_NewDataset.npy'))
      y_val = np.load(os.path.join(loc, 'Y_Val_Spec_NewDataset.npy'))
 
  # Otherwise, we will generate the vectors. Again, this takes a lot of time to do, so this is not recommended.
  else:
    x_train, y_train = DataGenerator(SAMPLING_RATE, NUM_MELS, FMAX, TOP_DB, EPS, audio_path, 'train') #### DO NOT RUN
    x_test, y_test = DataGenerator(SAMPLING_RATE, NUM_MELS, FMAX, TOP_DB, EPS, audio_path, 'test')
    x_val, y_val = DataGenerator(SAMPLING_RATE, NUM_MELS, FMAX, TOP_DB, EPS, audio_path, 'val')
    
    # If we are generating from the coswara_audio_path
    if audio_path == coswara_audio_path:
      np.save(os.path.join(loc, 'X_Train_Spec_Coswara'), x_train)
      np.save(os.path.join(loc, 'Y_Train_Spec_Coswara'), y_train)
      np.save(os.path.join(loc, 'X_Test_Spec_Coswara'), x_test)
      np.save(os.path.join(loc, 'Y_Test_Spec_Coswara'), y_test)
      np.save(os.path.join(loc, 'X_Val_Spec_Coswara'), x_val)
      np.save(os.path.join(loc, 'Y_Val_Spec_Coswara'), y_val)
    
    # If we are generating from the new_audio_path
    elif audio_path == new_audio_path:
      np.save(os.path.join(loc, 'X_Train_Spec_NewDataset'), x_train)
      np.save(os.path.join(loc, 'Y_Train_Spec_NewDataset'), y_train)
      np.save(os.path.join(loc, 'X_Test_Spec_NewDataset'), x_test)
      np.save(os.path.join(loc, 'Y_Test_Spec_NewDataset'), y_test)
      np.save(os.path.join(loc, 'X_Val_Spec_NewDataset'), x_val)
      np.save(os.path.join(loc, 'Y_Val_Spec_NewDataset'), y_val)
  
  return (x_train, y_train.reshape((-1, 1)), x_test, y_test.reshape((-1, 1)), x_val, y_val.reshape((-1, 1)))


def Data_Viz(n_row, n_cols, x_test, y_test):
  """
  Visualizer of the DataGenerator Function

  Parameters:
  ----------
  n_row: int
    Number of rows to show

  n_cols: int
    Number of columns to show

  x_test: np array
    Matrix of MFCC featuers

  y_test: np array
    Array of labels 1-positive, 0-negative

  Returns:
  --------
    N/A

  """
  assert (len(x_test.shape)==4), 'Input shape should be a N number of matrices'
  assert (len(y_test.shape)==2), 'Input shape should be a vector of shape (N, 1)'

  #Generate spectrograms for each audio sample with classification
  rows = n_row
  cols = n_cols
  n = rows*cols
  fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

  # Loops over a subset of the values in x_test and y_test
  # x_test are our spectrograms while y_test are our labels
  for i, (audio, status) in list(enumerate(zip(x_test, y_test)))[0:n]:
    r = i // cols
    c = i % cols
    ax = axes[r][c]

    
    plt.subplot(ax)
    librosa.display.specshow(audio.reshape(512, 215,))
    plt.colorbar()

    if status == 1:
      label = 'Positive'
    else:
      label = 'Negative'
    ax.set_title(label)
  plt.show()