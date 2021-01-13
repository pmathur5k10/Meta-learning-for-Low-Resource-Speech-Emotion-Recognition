import pandas as pd 
import librosa
import os 
import numpy as np
import sys
import pickle

'''
train_path_name = train_emo.iloc[:,0].tolist()
length_audio = []
for audio_path in train_path_name:
  x,sr = librosa.load(audio_path)
  mfccs = np.squeeze(np.asarray(librosa.feature.mfcc(x, sr=sr)))
  length_audio.append(np.shape(mfccs)[1])

print(np.max(length_audio))
print(np.mean(length_audio))

sys.exit()
'''

def extract_features(path):
  x,sr = librosa.load(path)
  ''' Returns shape (20,frames) '''
  mfccs = np.squeeze(np.asarray(librosa.feature.mfcc(x, sr=sr)))
  #mfccs = np.transpose( mfccs, (1, 0) )
  #return mfccs
  ''' Padding/Cutting to 120 frames'''
  if(np.shape(mfccs)[1]>=120): 
    return mfccs[:,:120]
  else:
    arr = np.zeros((np.shape(mfccs)[0],120))
    arr[:,:np.shape(mfccs)[1]] = mfccs
    return arr


def get_features(df):
  path_name = df.iloc[:,0].tolist()
  label = df.iloc[:,1].tolist()
  f_dict = {}
  for i, audio_path in enumerate(path_name):
    print(i, audio_path)
    audio_feature = extract_features(audio_path)
    f_dict[audio_path] = audio_feature
  return f_dict

#train_x , train_y = get_features(train_emo)
#val_x , val_y = get_features(val_emo)
#test_x , test_y = get_features(test_emo)

'''
with open('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/trainX.pkl','wb') as f:
     pickle.dump(train_x, f)

with open('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/trainY.pkl','wb') as f:
     pickle.dump(train_y, f)

with open('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/testX.pkl','wb') as f:
     pickle.dump(test_x, f)

with open('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/testY.pkl','wb') as f:
     pickle.dump(test_y, f)


with open('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/valX.pkl','wb') as f:
     pickle.dump(val_x, f)

with open('/gdrive/My Drive/IS_Meta_Learning/datasets/emo-db_files/valY.pkl','wb') as f:
     pickle.dump(val_y, f)
'''
