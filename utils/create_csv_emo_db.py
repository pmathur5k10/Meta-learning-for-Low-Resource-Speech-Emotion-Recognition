import glob
import pandas as pd
import os
from librosa_features import get_features
import sys
import pickle

data_path = './emo-db'

def write_emodb_csv(data_path, csv_name="emodb.csv",verbose=1):
    
    target = {"path": [], "emotion": []}
    categories = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }
    for file in glob.glob(data_path + "/wav/*.wav"):
        try:
            emotion = categories[os.path.basename(file)[5]]
        except KeyError:
            continue
        target['emotion'].append(emotion)
        target['path'].append(file)
        
        print (file, "Done")
        #sys.exit()

    if verbose:
        print("[EMO-DB] Total files to write:", len(target['path']))

    ''' 
    # dividing training/testing sets
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[EMO-DB] Training samples:", train_size)
        print("[EMO-DB] Testing samples:", test_size)   
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
    df = write_emodb_csv(data_path)
    f_dict = get_features(df)
    with open('./emo-db/emodb.pkl', 'wb') as f:
       pickle.dump(f_dict, f)