"""
Metric API
"""
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot
from numpy import argmax
import tensorflow as tf
import numpy as np

metric = {1:'sensitivity_specitivity', 2:'precision_recall', 3:'f_score'}

################# METRIC API ###################################

def round_using_t(prediction, threshold):

  """
  Returns labels that are greater than the threshold

  Parmeters
  ---------
  prediction: ndarray
    Predicted labels

  threshold: int
    Threshold to use

  Returns
  -------
    Labels greater than the threshold
  
  """
  return (prediction >= threshold).astype('int')

def get_best_threshold(model, x_test, y_test, metric, plot=True):

  """
  Determines best threshold based on metric to use
  
  Parameters
  ----------
  model: Tensorflow Model
    Model Created

  x_test: ndarray
    Testing dataset, x_values

  y_test: ndarray
    Testing dataset, y_values

  metric: str
    Used to determine which metric we want to maximize

  plot: bool
    True: Plot
    False: No plot

  Returns
  -------
  y_hat: ndarray
    Predicted labels

  theshold:
    Threshold estimated to maximize metric

  score:
    Value best threshold maximizes

  """

  #prediction = model(x_test, training=False)
  prediction = model.predict(x_test)
  proto_tensor = tf.make_tensor_proto(prediction)
  y_hat = tf.make_ndarray(proto_tensor)
  y_hat = y_hat.reshape(-1,)

  if metric == 'sensitivity_specitivity':
    
    fpr, tpr, thresholds = roc_curve(y_test.reshape(-1,), y_hat)
    gmeans = np.sqrt(tpr * (1-fpr)) # The Geometric Mean or G-Mean is a metric for imbalanced classification that, if optimized, will seek a balance between the sensitivity and the specificity.
    ix = argmax(gmeans)
    score = gmeans[ix]

    print('AUC:', auc(1-fpr, tpr))
    
    if plot:
      pyplot.plot(1-fpr, tpr, marker='.')
      pyplot.scatter(1-fpr[ix], tpr[ix], marker='o', color='black', label='Optimal Threshold')
      pyplot.xlabel('Specificity', fontsize=16)
      pyplot.ylabel('Sensitivity', fontsize=16)
      pyplot.legend(loc='upper left')
      pyplot.title('Sensitivity-Specitivity Curve', fontsize=20)
      pyplot.show()
  
  elif metric == 'precision_recall':
    
    precision, recall, thresholds = precision_recall_curve(y_test.reshape(-1,), y_hat)
    fscore = (2 * precision * recall) / (precision + recall) # If we are interested in a threshold that results in the best balance of precision and recall, then this is the same as optimizing the F-measure that summarizes the harmonic mean of both measures.
    ix = argmax(fscore)
    score = fscore[ix]
    print('AUC:', auc(recall, precision))

    if plot:
      pyplot.plot(recall, precision, marker='.')
      pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Optimal Threshold')
      pyplot.xlabel('Recall', fontsize=16)
      pyplot.ylabel('Precision', fontsize=16)
      pyplot.legend(loc='upper left')
      pyplot.title('Precision-Recall Curve', fontsize=20)
      pyplot.show()

  else:
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(y_test.reshape(-1,), round_using_t(y_hat, t)) for t in thresholds]
    ix = argmax(scores)
    score = scores[ix]

  return y_hat, thresholds[ix], score

def get_predict(prediction, threshold):

  """
  Returns the labels rounded using the threshold given

  Parameters
  ---------
  prediction: ndarray
    predicted labels

  threshold: int
    threshold to use for rounding

  Returns
  -------
  prediction: ndarray
    rounded prediction
  """

  prediction[prediction < threshold] = 0
  prediction[prediction >= threshold] = 1
  
  return prediction

def get_sen_spec(y_test, y_hat):
  """
  Get the final sensitivity and specificity of predicted labels

  Parameters
  ----------
  y_test: ndarray
    True labels

  y_hat: ndarray
    Predicted labels
  Returns
  -------
  sensitivity: int
    Sensitivity value

  specificity: int
    Specificity value
  """
  cm = confusion_matrix(y_test.reshape(-1,), y_hat)
  sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
  #print('Sensitivity : ', sensitivity)
  specificity = cm[1,1]/(cm[1,0]+cm[1,1])
  #print('Specificity : ', specificity)
  return sensitivity, specificity
