# -*- coding: utf-8 -*-
"""Parameters

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YC5JRyMZCHNWbpTAO6Fq9OmZhuU6M1MK
"""

# Sampling Rate
SAMPLING_RATE = 22000
# Duration of the signal in seconds
DURATION = 5
# Length of the FFT window
NUM_MELS = 512
# The maximum frequency
FMAX = 10000
# The threshold (in decibels) below reference to consider as silence
TOP_DB = 30
# Variance
EPS = 1e-6
# Seed to be used in random.shuffle()
SEED = 42
# Number of MFC Coefficients
N_MFCC = 13
# Number of PCA Components used in feature reduction
PCA_COMPONENTS = 256

# Listed used in the Surfboard API to produce the handcrafted feature vector
components_list = ['mfcc', 'log_melspec', 'morlet_cwt', 'spectral_skewness', 'spectral_kurtosis', 'spectral_rolloff', 'rms']
statistics_list = ['mean', 'std', 'skewness', 'kurtosis', 'first_derivative_mean', 'first_derivative_std', 'first_derivative_skewness', 'first_derivative_kurtosis']