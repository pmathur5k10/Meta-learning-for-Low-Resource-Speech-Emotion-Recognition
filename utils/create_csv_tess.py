import glob
import pandas as pd
import os
import sys
import pickle
from librosa_features import get_features

data_path = './TESS'
audio_featDict = {}

def write_savee_csv(data_path, csv_name="tess.csv", verbose=1):
    target = {"path": [], "emotion": []}

    categories = {
    "disgust": "disgust", 
    "angry": "angry",
    "fear": "fear",
    "happy": "happy", 
    "neutral": "neutral",
    "ps": "pleasant_surprise",
    "sad": "sad"
    }

    root = data_path

    for r, _ ,files in os.walk(root):
        for file in files:
            if not file.endswith('.wav'):
                continue 
            
            file_path = os.path.join( r, file )
            try:
                f = file_path[:-len('.wav')]
                lbl = f.split('_')[-1]
                emotion = categories[lbl]
            except KeyError:
               print ("Error", file_path)
               continue

            target['emotion'].append(emotion)
            target['path'].append(file_path)

            print (file_path, "Done")

    if verbose:
        print("[TESS-DB] Total files to write:", len(target['path']))

    '''    
    #dividing training/testing sets
    n_samples = len(target['path'])
    test_size = int((test_ratio) * n_samples)
    train_size = int((1-test_ratio) * n_samples)

    if verbose:
        print("[RAVDESS-DB] Training samples:", train_size)
        print("[RAVDESS-DB] Testing samples:", test_size)   
    X_train = target['path'][:train_size]
    X_test = target['path'][train_size:]
    y_train = target['emotion'][:train_size]
    y_test = target['emotion'][train_size:]
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)
    '''
    df = pd.DataFrame({"path": target['path'], "emotion": target['emotion']})
    df.to_csv(csv_name, index = False)
    return df

if __name__ == '__main__':
    df = write_savee_csv(data_path)
    f_dict = get_features(df)
    with open(data_path +  '/tess.pkl', 'wb') as f:
       pickle.dump(f_dict, f)