import random
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
import numpy as np
import sys
import logging

emotion_list = ["angry","sad","happy","neutral"] 
datasets_train = ['tess','emodb','ravdess','iemocap']
datasets_test = ['emovo', 'shemo', 'urdu']

pkl_paths = [ os.path.join('../data', data, data + '.pkl') for data in datasets_train ]
pkl_paths += [ os.path.join('../data', data, data + '.pkl') for data in datasets_test ]

train_csv_paths = [ os.path.join('../data', data, data + '_train.csv') for data in datasets_train ]
train_csvs = [ pd.read_csv(path) for path in train_csv_paths ]

test_csv_paths = [ os.path.join('../data', data, data + '_test.csv') for data in datasets_test ]
test_csvs = [ pd.read_csv(path) for path in test_csv_paths ]

pkls = []
for p in pkl_paths:
    with open(p, "rb") as f:
        audio_feat = pickle.load(f)
    pkls.append( audio_feat )

train_df = pd.concat(train_csvs)
test_df = pd.concat(test_csvs)
audio_feat = {} 
for d in pkls: audio_feat.update(d)

def split_emotions(SEED):
 
    random.seed(SEED)
    global train_df, test_df

    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_chars = train_df
    test_chars = test_df

    return train_chars, test_chars

class Task(object):
    """
    An abstract class for defining a single few-shot task.
    """

    def __init__(self, args, df, num_classes, support_num, query_num):
        """
        train_* are a support set
        test_* are a query set
        meta_* are for meta update in meta-learner
        Args:
            character_folders: a list of omniglot characters that the task has
            num_classes: a number of classes in a task (N-way)
            support_num: a number of support samples per each class (K-shot)
            query_num: a number of query samples per each class NOTE how to configure ??
        """
        self.df = df
        self.num_classes = num_classes
        self.support_num = support_num
        self.query_num = query_num

        class_folders = emotion_list
        labels = list(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))

        samples = dict()

        self.train_roots = []
        self.test_roots = []
        self.train_labels = []
        self.test_labels = []

        for c in class_folders:

            temp = self.df[ self.df['emotion'] == c ]
            temp = list( temp['path'] )
            samples[c] = random.sample(temp, len(temp))
            self.train_roots += samples[c][:support_num]
            self.train_labels += [ labels[c] for i in range(support_num) ]

            self.test_roots += samples[c][support_num:support_num + query_num]
            self.test_labels += [ labels[c] for i in range(query_num) ]

        samples = dict()
        self.meta_roots = []
        self.meta_labels = []
        for c in class_folders:
            temp = self.df[ self.df['emotion'] == c ]
            temp = list( temp['path'] )
            samples[c] = random.sample(temp, len(temp))
            self.meta_roots += samples[c][:support_num]
            self.meta_labels += [ labels[c] for i in range(support_num) ]

class FewShotDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, filenames, labels, num_classes, frame_length=120):
        """
        Store the filenames of the images to use.
        Specifies transforms to apply on images.

        Args:
            filenames: (list) a list of filenames of images in a single task
            labels: (list) a list of labels of images corresponding to filenames
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = filenames
        self.labels = labels
        self.audio_feat = audio_feat
        self.num_classes = num_classes
        self.frame_length = frame_length

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        ft = self.audio_feat[ self.filenames[idx] ]

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

        return ft.float(), self.labels[idx]

class SER(Task):
    """
    Class for defining a single few-shot task given ImageNet dataset.
    """

    def __init__(self, *args, **kwargs):
        super(SER, self).__init__(*args, **kwargs)

def fetch_dataloaders(types, task, params):
    """
    Fetches the DataLoader object for each type in types from task.
    TODO for MAML

    Args:
        types: (list) has one or more of 'train', 'val', 'test' 
               depending on which data is required # TODO 'val'
        task: (OmniglotTask or TODO ImageNet) a single task for few-shot learning
        TODO params: (Params) hyperparameters
    Returns:
        dataloaders: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    for split in ['train', 'val', 'test', 'meta']:
        if split in types:
            # use the train_transformer if training data,
            # else use eval_transformer without random flip
            if split == 'train':
                train_filenames = task.train_roots
                train_labels = task.train_labels
                dl = DataLoader(
                    FewShotDataset(train_filenames, train_labels, params.num_classes),
                    batch_size=len(train_filenames),  # full-batch in episode
                    shuffle=True)  # TODO args: num_workers, pin_memory
            elif split == 'test':
                test_filenames = task.test_roots
                test_labels = task.test_labels
                dl = DataLoader(
                    FewShotDataset(test_filenames, test_labels, params.num_classes),
                    batch_size=len(test_filenames),  # full-batch in episode
                    shuffle=False)
            elif split == 'meta':
                meta_filenames = task.meta_roots
                meta_labels = task.meta_labels
                dl = DataLoader(
                    FewShotDataset(meta_filenames, meta_labels, params.num_classes),
                    batch_size=len(meta_filenames),  # full-batch in episode
                    shuffle=True)  # TODO args: num_workers, pin_memory
            else:
                # TODO
                raise NotImplementedError()
            dataloaders[split] = dl

    return dataloaders