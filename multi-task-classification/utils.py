import numpy as np
import os
import cv2
import json
import torch
import torchvision
import pandas as pd
import pickle
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BaseDataLoader:
    def __init__(self, batch_size=1, train=True, shuffle=True, drop_last=False):
        pass

    def get_loader(self, loader, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
        raise NotImplementedError

class InputDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, filenames, labels, num_classes, audio_feat, emotion_list, frame_length=120):
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
        self.emotions = list(range(len(emotion_list)))
        self.label_dict = dict(zip(emotion_list, self.emotions))
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

        return ft.float(), self.label_dict[ self.labels[idx] ]

class SERLoader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True, drop_last=True, fraction = 1):
        super(SERLoader, self).__init__(batch_size, train, shuffle, drop_last)
        
        self.emotion_list = ["angry","sad","happy","neutral"] 
        self.datasets_train = ['tess', 'savee', 'iemocap', 'ravdess', 'urdu']
        self.datasets_test = ['urdu']
        
        self.train_csv_paths = [ os.path.join('../data', data, data + '_train.csv') for data in self.datasets_train ]
        self.train_csvs = [ pd.read_csv(path) for path in self.train_csv_paths ]

        num_samples = fraction * len( self.train_csvs[-1] )
        self.train_csvs[-1] = self.train_csvs[-1][:num_samples]

        self.test_csv_paths = [ os.path.join('../data', data, data + '_test.csv') for data in self.datasets_test ]
        self.test_csvs = [ pd.read_csv(path) for path in self.test_csv_paths ]
        
        self.pkl_paths = [ os.path.join('../data', data, data + '.pkl') for data in self.datasets_test ]
        if train:
            self.pkl_paths = [ os.path.join('../data', data, data + '.pkl') for data in self.datasets_train ]

        self.num_classes = len( self.emotion_list )

        pkls = []
        for p in self.pkl_paths:
            with open(p, "rb") as f:
                audio_feat = pickle.load(f)
            pkls.append( audio_feat )

        self.df = None
        self.train_df = pd.concat(self.train_csvs)
        self.test_df = pd.concat(self.test_csvs)

        self.train_df = self.train_df[ self.train_df['emotion'].isin(self.emotion_list) ]
        self.test_df = self.test_df[ self.test_df['emotion'].isin(self.emotion_list) ]

        if train:
            self.df = self.train_df
        else:
            self.df = self.test_df

        self.audio_feat = {} 
        for d in pkls: self.audio_feat.update(d)

        dataset = InputDataset( list(self.df['path']), list(self.df['emotion']), 
        self.num_classes, self.audio_feat, self.emotion_list )

        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)
        self.task_dataloader = None

        self._len = len(self.df)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


    def _create_TaskDataLoaders(self):
        images = []
        labels = []

        for batch_images, batch_labels in self.dataloader:
            for i in batch_images:
                images.append(i)
            for l in batch_labels:
                labels.append(l)

        self.task_dataloader = []
        for t in range(self.num_classes):
            dataset = CustomDataset(data=images.copy(), labels=[(c == t).long() for c in labels])
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)
            self.task_dataloader.append(dataloader)


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if self.task_dataloader is None:
            self._create_TaskDataLoaders()

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(self.num_classes)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(self.num_classes))
        else:
            assert task in list(range(self.num_classes)), 'Unknown task: {}'.format(task)
            labels = [0 for _ in range(self.num_classes)]
            labels[task] = 1
            return labels


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes_single(self):
        return 4


    @property
    def num_classes_multi(self):
        return [2 for _ in range(4)]

class MultiTaskDataLoader:
    def __init__(self, dataloaders, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task

def plot_training_results(model_dir, plot_history):
    """
    Plot training results (procedure) during training.

    Args:
        plot_history: (dict) a dictionary containing historical values of what 
                      we want to plot
    """
    te_accs = plot_history['test_acc']
    te_f1 = plot_history['test_f1']

    plt.figure(0)
    plt.plot(list(range(len(te_accs))), te_accs, label='test_acc')
    plt.title('Accuracy trend')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'accuracy_trend'), dpi=200)

    plt.figure(1)
    plt.plot(list(range(len(te_f1))), te_f1)
    plt.xlabel('episode')
    plt.ylabel('f1')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'test_f1'), dpi=200)

    plt.close(0)
    plt.close(1)

if __name__ == '__main__':
    SER = SERLoader()
    l = SER.get_loader(loader='standard')
    for x, y in l:
        print (x.shape, y.shape)
        break