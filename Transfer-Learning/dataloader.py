import os
import torch
import pandas as pd
import pickle
import numpy as np

import os
import torch
import pandas as pd
import pickle
import numpy as np


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, path_csvs, feature_pkls, frame_length=120):
        self.csvs = [ pd.read_csv(path) for path in path_csvs ]
        self.df = pd.concat(self.csvs)

        pkls = []
        for p in feature_pkls:
            with open(p, "rb") as f:
                audio_feat = pickle.load(f)
            pkls.append( audio_feat )
        
        self.audio_feat = {} 
        for d in pkls: self.audio_feat.update(d)

        self.emotions = {
            "angry": 0,
            "happy": 1,
            "neutral": 2,
            "sad": 3,
        }

        self.df = self.df[ self.df['emotion'].isin(self.emotions.keys()) ]
        self.df =  self.df.reset_index()
        self.num_emotions = len(self.emotions.keys())
        self.frame_length =  frame_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = self.df.loc[idx, 'path']
        emotion = self.df.loc[idx, 'emotion']

        ft = self.audio_feat[ audio_name ]

        if ft.shape[1] != 20: 
            ft = np.transpose( ft, (1, 0) )

        if ft.shape[0] > self.frame_length:
            ft = ft[:self.frame_length, :]
        elif ft.shape[0] < self.frame_length:
            ft = np.block([
                [ft],
                [np.zeros((self.frame_length - ft.shape[0], ft.shape[1]))]
            ])

        ft = torch.from_numpy(ft)

        return ft.float(), self.emotions[ emotion ]


if __name__ == '__main__':
    ds = DataLoader(path_csvs=['../data/emovo/emovo_train.csv'], feature_pkls=['../data/emovo/emovo.pkl'])
    x, y = ds.__getitem__(5)
    print (x.shape)