import surfboard
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
import librosa
import librosa.display
import IPython.display as ipd
import tensorflow as tf
from tqdm import tqdm
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from utils.Commented.DataGenerator import DataGenerator, Data_Viz, Get_matrix_inputs
from utils.Commented.SurfboardFeatures import GenerateVectors, Reduce_Dim, Get_feature_vectors
from utils.Commented.Modeling import build_model2, build_model, build_cnn_train, build_train
from utils.Commented.Plotter import loss_plot, accuracy_plot, plot_cm
from utils.Commented.Metrics import round_using_t, get_best_threshold, get_predict, get_sen_spec
from utils.Commented.Transfer_Learning import expand_dim, expand_dim_all, make_transfer_model
from utils.Commented import Parameters
import os

os.chdir('/content/gdrive/MyDrive/DSCI400')
new_audio_path = '/content/gdrive/MyDrive/DSCI400/Regroup Audio New/coughvid'
coughnet_audio_path = '/content/gdrive/MyDrive/DSCI400/Coughnet Audio'
coswara_audio_path = '/content/gdrive/MyDrive/DSCI400/Regroup Audio New/cough-coswara'
audio_path = new_audio_path
metric = {1:'sensitivity_specitivity', 2:'precision_recall', 3:'f_score'}

def CNN_Model(train = False, load = True, report = True, show_plots = False):

  """
  Full process for CNN model
  Parameters
  ---------
  train: bool
    whether to train or return model
  load: bool
    load weights
  report: bool
    report performace
  show_plots: bool
    show plots of the model summary
  Returns
  ------
  if train is true, returns history, else returns model
  """
  x_train, y_train, x_test, y_test, x_val, y_val = Get_matrix_inputs(new_audio_path, coswara_audio_path, coswara_audio_path, Parameters.saved_load_loc, load=True)
  input_shape_1 = x_train.shape[1:]
  a = None
  a = build_model(input_shape_1, 2)

  if train:
    cnn_hist = build_cnn_train(a, x_train, y_train, Parameters.BATCH_SIZE, Parameters.EPOCHS, Parameters.LEARNING_RATE, Parameters.checkpoint_loc, 'test2', show_plots, x_val, y_val)
    if report:
      predict, threshold, score = get_best_threshold(a, x_test, y_test, metric[1], report)
      y_hat = get_predict(predict, threshold)
      tn, fp, fn, tp = plot_cm(y_test.reshape(-1,), y_hat, threshold)
      Sensitivity, Specificity = get_sen_spec(y_test, y_hat)
      report = classification_report(y_test.reshape(-1,), y_hat, target_names=['0', '1'])
      print('Threshold:', threshold, 'Score:', score)
      print('true_neg:', tn)
      print('false_pos:', fp)
      print('false_neg:', fn)
      print('true_pos:', tp)
      print('Sensitivity : ', Sensitivity)
      print('Specificity : ', Specificity)
      print(report)
    return cnn_hist
  else:
    return a