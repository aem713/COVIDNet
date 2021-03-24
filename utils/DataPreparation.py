import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import tensorflow as tf


def create_mel(path, sr = 22000, n_mels=512, fmax = 10000, top_db=30, eps = 1e-6, plot = False):

  y , sr = librosa.load(path, sr=sr)
  yt, index = librosa.effects.trim(y, top_db=top_db)

  if yt.shape[0] < 5*sr:
    yt=np.pad(yt,int(np.ceil((5*sr-yt.shape[0])/2)), mode='reflect')
  else:
    yt=yt[:5*sr]

  S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=n_mels, fmax=fmax)
  S_dB = librosa.power_to_db(S, ref=np.max)
  
  if plot:
    plt.figure()
    librosa.display.specshow(S_dB)
    plt.colorbar()

  mean = S_dB.mean()
  std = S_dB.std()
  S_dB_norm = (S_dB - mean) / (std + eps)
  S_dB_min, S_dB_max = S_dB_norm.min(), S_dB_norm.max()
  S_dB_scaled = 255 * (S_dB_norm - S_dB_min) / (S_dB_max - S_dB_min)
  S_dB_scaled = S_dB_scaled.astype(np.uint8)
  
  if plot:
    plt.figure()
    librosa.display.specshow(S_dB_scaled)
    plt.colorbar()

  return S_dB_scaled


class DataGenerator():
    def __init__(self, sr, n_mels, fmax, top_db, eps, seed, path, dataset_type="train"):
        
        self.path = path
        self.sr = sr
        self.n_mels = n_mels
        self.fmax = fmax
        self.top_db = top_db
        self.eps = eps
        self.seed = seed

        positive_paths = []
        negative_paths = []
        for entry in os.scandir(os.path.join(self.path, os.path.join(dataset_type, 'pos'))):
            if entry.is_file():
                positive_paths.append(entry.path)
        for entry in os.scandir(os.path.join(self.path, os.path.join(dataset_type, 'neg'))):
            if entry.is_file():
                negative_paths.append(entry.path)
        
        path_to_audio_p = list(map(lambda x: create_mel(x, self.sr, self.n_mels, self.fmax, self.top_db, self.eps, plot=False), positive_paths))
        pos_audio_ds = tf.data.Dataset.from_tensor_slices(path_to_audio_p)
        pos_labels_ds = tf.data.Dataset.from_tensor_slices([1 for i in positive_paths])
        p_ds = tf.data.Dataset.zip((pos_audio_ds, pos_labels_ds))

        path_to_audio_n = list(map(lambda x: create_mel(x, self.sr, self.n_mels, self.fmax, self.top_db, self.eps, plot=False), negative_paths))
        neg_audio_ds = tf.data.Dataset.from_tensor_slices(path_to_audio_n)
        neg_labels_ds = tf.data.Dataset.from_tensor_slices([0 for i in negative_paths])
        n_ds = tf.data.Dataset.zip((neg_audio_ds, neg_labels_ds))
        
        combined_ds = p_ds.concatenate(n_ds)
        combined_ds = combined_ds.shuffle(len(combined_ds), seed=self.seed)
        self.data = combined_ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return list(self.data.as_numpy_iterator())[index]