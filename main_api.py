import librosa

def create_mel(path, sr = 22000, n_mels=512, fmax = 10000, top_db=30, eps = 1e-6):

  y , sr = librosa.load(path, sr=sr)
  yt, index = librosa.effects.trim(y, top_db=top_db)

  if yt.shape[0] < 5*sr: # 5*22000 = 110,000
    yt=np.pad(yt,int(np.ceil((5*sr-yt.shape[0])/2)), mode='reflect') # np.pad actually pads to both sides of yt. So we pad half of the time between 5 seconds and the length of yt before AND after yt a reflected version of yt
  else:
    yt=yt[:5*sr]

  S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=n_mels, fmax=fmax)
  S_dB = librosa.power_to_db(S, ref=np.max)

  #plt.figure()
  #librosa.display.specshow(S_dB)
  #plt.colorbar()

  mean = S_dB.mean()
  std = S_dB.std()
  S_dB_norm = (S_dB - mean) / (std + eps)
  S_dB_min, S_dB_max = S_dB_norm.min(), S_dB_norm.max()
  S_dB_scaled = 255 * (S_dB_norm - S_dB_min) / (S_dB_max - S_dB_min)
  S_dB_scaled = S_dB_scaled.astype(np.uint8)

  #plt.figure()
  #librosa.display.specshow(S_dB_scaled)
  #plt.colorbar()

  return S_dB_scaled
