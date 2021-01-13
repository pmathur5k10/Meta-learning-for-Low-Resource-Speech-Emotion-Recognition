import glob
import pandas as pd
import os
from librosa_features import get_features
import sys
import pickle
import numpy as np

data_path = './IEMOCAP'

def write_iemocap_csv(data_path, csv_name="iemocap.csv",verbose=1):
    
    target = {"path": [], "emotion": []}

    categories = {
        'ang': 'angry', 
        'hap': 'happy', 
        'exc': 'excitement',
        'sad': 'sad',
        'fru': 'frustration', 
        'fea': 'fear',
        'sur': 'surprise',
        'neu': 'neutral', 
        'oth': 'other',
        'dis': 'disgust',
    }

    data = []
    for i in range(1, 6):
        # Define path to evaluation files of this session
        path = os.path.join(data_path, 'Session' + str(i), 'dialog', 'EmoEvaluation')

        # Get list of evaluation files
        files = [file for file in os.listdir(path) if file.endswith('.txt')]

        # Iterate through evaluation files to get utterance-level data
        for file in files:
            # Open file
            f = open(os.path.join(path, file), 'r')

            # Get list of lines containing utterance-level data. Trim and split each line into individual string elements.
            data += [line.strip()
                            .replace('[', '')
                            .replace(']', '')
                            .replace(' - ', '\t')
                            .replace(', ', '\t')
                            .split('\t')
                        for line in f if line.startswith('[')]

    data = [d + [d[2][4], d[2].split('_')[1], d[2][-4], d[2][-3:]] for d in data]

    for i in range( len(data) ):
        d = data[i]
        if d[3] == 'xxx':
            continue
        target['path'].append( d[2] )
        target['emotion'].append( categories[ d[3] ] )

    target['path'] = [ os.path.join(data_path, 'Session' + file[4], 'sentences', 'wav', file[:-5], file + '.wav') for file in target['path'] ]

    if verbose:
        print("[IEMOCAP-DB] Total files to write:", len(target['path']))

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
    df = write_iemocap_csv(data_path)
    f_dict = get_features(df)
    with open(data_path + '/iemocap.pkl', 'wb') as f:
       pickle.dump(f_dict, f)