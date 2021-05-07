# -*- coding: utf-8 -*-
"""
Plotter API
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def loss_plot(model, model_label, val=False):
  """
  Creates a plot of the loss function of the model
  Parameters
  ----------
  model: tf.model
    the model to plot
  
  model_label: str
    label of the model
  
  val: bool
    used in case validation is provided

  Returns
  -------
  None
  """
  plt.plot(model.history['loss'], label='Training Loss')
  if val:
    plt.plot(model.history['val_loss'], label='Validation Loss')
  plt.title('Loss Function for' + model_label + 'Model', fontsize=20)
  plt.ylabel('Loss Value', fontsize=16)
  plt.xlabel('No. epoch', fontsize=16)
  plt.legend(loc="upper left", fontsize=16)
  plt.show()

def accuracy_plot(model,model_label, val=False):
  """
  Creates a plot of the accuracy function of the model
  Returns
  -------
  model: tf.model
    the model to plot
  
  model_label: str
    label of the model
  
  val: bool
    used in case validation is provided
  Returns
  -------
  None
  """
  plt.plot(model.history['accuracy'], label='Accuracy (training data)')
  if val:
    plt.plot(model.history['val_accuracy'], label='Accuracy (Validation data)')
  plt.title('Accuracy for' + model_label + 'Model', fontsize=20)
  plt.ylabel('Accuracy Value', fontsize=16)
  plt.xlabel('No. epoch', fontsize=16)
  plt.legend(loc="upper left")
  plt.show()

def plot_cm(labels, predictions, threshold):

  """
  Plot the confusion matrix

  Parameters
  ----------
  labels: ndarray
    True labels

  predictions: ndarray
    Predicted labels

  threshold: int
    threshold set to round predictions into 0 or 1

  Returns
  -------
  None

  """
  cm = confusion_matrix(labels, predictions > threshold)
  tn, fp, fn, tp = confusion_matrix(labels, predictions> threshold).ravel()
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix with Threshold of {:.4f}'.format(threshold), fontsize=20)
  plt.ylabel('Actual label', fontsize=16)
  plt.xlabel('Predicted label', fontsize=16)
  return tn, fp, fn, tp