import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd


# split valid_set from train_set for IEMOCAP and MELD dataset
def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


class IEMOCAPDataset(Dataset):
    def __init__(self, path=None, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        # label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        # label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


def get_MELD_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path=path)
    testset = MELDDataset(path=path, train=False)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class M3EDDataset_New(Dataset):
    def __init__(self, path, train_path, valid_path, test_path, train=True, valid=False, test=False):
        self.videoIDs, self.Speakers, self.Labels, self.Text, self.Audio, self.Vision, self.Sentence, \
        _, _, _ = pickle.load(open(path, 'rb'), encoding='latin1')
        # label index mapping = {'hap':0, 'neu':1, 'sad':2, 'dis':3, 'ang':4, 'fea':5, 'sur':6}  seven emotion
        self.train_ids = self.get_ids(train_path)
        self.valid_ids = self.get_ids(valid_path)
        self.test_ids = self.get_ids(test_path)
        if train:
            self.keys = [x for x in self.train_ids]
        elif valid:
            self.keys = [x for x in self.valid_ids]
        elif test:
            self.keys = [x for x in self.test_ids]
        else:
            raise AttributeError('train, valid, test at least one true, the another two false')
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.Text[vid]), \
               torch.FloatTensor(self.Audio[vid]), \
               torch.FloatTensor(self.Vision[vid]), \
               torch.FloatTensor([[1, 0] if x == 'A' else [0, 1] for x in self.Speakers[vid]]), \
               torch.FloatTensor([1] * len(self.Labels[vid])), \
               torch.LongTensor(self.Labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def get_ids(self, id_path):
        f = open(id_path, mode='r', encoding='utf-8')
        split_ids = []
        for line in f:
            split_id = line.replace('\n', '')
            # split_id = line.rstrip('\n')
            split_ids.append(split_id)
        return split_ids

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


def get_new_M3ED_loaders(path, train_path, valid_path, test_path, batch_size=32, num_workers=0, pin_memory=False):
    train_set = M3EDDataset_New(path, train_path, valid_path, test_path, train=True, valid=False, test=False)
    valid_set = M3EDDataset_New(path, train_path, valid_path, test_path, train=False, valid=True, test=False)
    test_set = M3EDDataset_New(path, train_path, valid_path, test_path, train=False, valid=False, test=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=train_set.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=valid_set.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=test_set.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

