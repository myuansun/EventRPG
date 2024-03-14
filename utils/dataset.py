import numpy as np
from os import listdir
from os.path import join
import torch
from torch.utils.data.dataloader import default_collate
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import split_to_train_test_set

def get_dataset(args):
    if args.dataset == "NCaltech101":
        args.train_dataset = "..../N-Caltech101/training"
        args.validation_dataset = "..../N-Caltech101/validation"
        args.test_dataset = "..../N-Caltech101/testing"
        if hasattr(args, "spiking_neuron"):
            train_ds, test_ds = NCaltech101_Resplit(args.train_dataset, args.validation_dataset, args.test_dataset)
            val_ds = test_ds
        else:
            train_ds = NCaltech101(args.train_dataset)
            val_ds = NCaltech101(args.validation_dataset)
            test_ds = NCaltech101(args.test_dataset)
    elif args.dataset == "CIFAR10DVS":
        dataset = "..../CIFAR10DVS"
        train_ds, test_ds = Cifar10DVS(dataset)
        val_ds = test_ds
    elif args.dataset == "NCars":
        args.train_dataset = "..../N-Cars/train"
        args.test_dataset = "..../N-Cars/test"
        train_ds = NCars(args.train_dataset)
        test_ds = NCars(args.test_dataset)
        val_ds = test_ds
    elif args.dataset == "DVSGesture":
        dataset = "..../dataset/DVSGesture"
        train_ds, test_ds = DVSGesture(dataset)
        val_ds = test_ds
    elif args.dataset == "SLAnimals3sets":
        args.train_dataset = "..../SLAnimals/SL_animal_splits/dataset_3sets_2000/train"
        args.test_dataset = "..../SLAnimals/SL_animal_splits/dataset_3sets_2000/test"
        train_ds = SLAnimals(args.train_dataset)
        test_ds = SLAnimals(args.test_dataset)
        val_ds = test_ds
    elif args.dataset == "SLAnimals4sets":
        args.train_dataset = "..../SLAnimals/SL_animal_splits/dataset_4sets_2000/train"
        args.test_dataset = "..../SLAnimals/SL_animal_splits/dataset_4sets_2000/test"
        train_ds = SLAnimals(args.train_dataset)
        test_ds = SLAnimals(args.test_dataset)
        val_ds = test_ds
    elif args.dataset == "miniNImageNet":
        args.train_dataset = "..../miniNImageNet/extracted_train"
        args.test_dataset = "..../miniNImageNet/extracted_val"
        train_ds = miniNImageNet(args.train_dataset)
        test_ds = miniNImageNet(args.test_dataset)
        val_ds = test_ds
    else:
        raise Exception
    return train_ds, val_ds, test_ds

class NCaltech101:
    def __init__(self, root):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)
        self.np_labels = np.array(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        events[:, 3] = (events[:, 3] + 1) / 2
        return torch.from_numpy(events), label

    def sample_labeled_data(self, label, bs):
        idx = np.where(self.np_labels == label)[0]
        if len(idx) >= bs:
            # np.random.shuffle(idx)
            shuffle_idx = idx[:bs]
        else:
            bs = len(idx)
            shuffle_idx = idx
        files, labels = zip(*[self[i] for i in shuffle_idx])
        files = list(files)
        return files, bs


def NCaltech101_Resplit(train_root, val_root, test_root):
        classes = listdir(train_root)
        classes.sort()
        train_files = []
        train_labels = []
        test_files = []
        test_labels = []
        for i, c in enumerate(classes):
            new_files = [join(train_root, c, f) for f in listdir(join(train_root, c))]
            train_files += new_files
            train_labels += [i] * len(new_files)
        for i, c in enumerate(classes):
            new_files = [join(val_root, c, f) for f in listdir(join(val_root, c))]
            train_files += new_files
            train_labels += [i] * len(new_files)
        for i, c in enumerate(classes):
            new_files = [join(test_root, c, f) for f in listdir(join(test_root, c))]
            train_len = int(len(new_files) / 2 + 1)
            test_len = len(new_files) - train_len
            train_files += new_files[:train_len]
            train_labels += [i] * train_len
            test_files += new_files[train_len:]
            test_labels += [i] * test_len
        train_set = NCaltech101_resplit(train_files, train_labels, classes)
        test_set = NCaltech101_resplit(test_files, test_labels, classes)
        return train_set, test_set


class NCaltech101_resplit:
    def __init__(self, files, labels, classes):
        self.files = files
        self.labels = labels
        self.classes = classes
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        events[:, 3] = (events[:, 3] + 1) / 2
        return torch.from_numpy(events), label


class SLAnimals:
    def __init__(self, root, resolution=(128, 128)):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        self.resolution = resolution

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        return torch.from_numpy(events), label

class NCars:
    def __init__(self, root, resolution=(120, 100)):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        self.resolution = resolution

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        events[:, 3] = (events[:, 3] + 1) / 2
        return torch.from_numpy(events), label

def Cifar10DVS(root, split_ratio=0.9):
    dataset = CIFAR10DVS(root, data_type="event")
    train_set, test_set = split_to_train_test_set(split_ratio, dataset, num_classes=10, random_split=True)
    return SpikingjellyDataset(train_set), SpikingjellyDataset(test_set)

class miniNImageNet:
    def __init__(self, root, resolution=(224, 224)):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        self.resolution = resolution

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        file = np.load(f)['event_data']
        events = np.concatenate([file['x'][:, np.newaxis], file['y'][:, np.newaxis], file['t'][:, np.newaxis], file['p'][:, np.newaxis]], axis=1).astype(np.float32)
        return torch.from_numpy(events), label

def DVSGesture(root):
    train_set = DVS128Gesture(root, train=True, data_type="event")
    test_set = DVS128Gesture(root, train=False, data_type="event")
    return SpikingjellyDataset(train_set), SpikingjellyDataset(test_set)


class SpikingjellyDataset:

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dict_events, label = self.dataset[idx]
        x = dict_events['x'].astype(np.float32)
        y = dict_events['y'].astype(np.float32)
        t = dict_events['t'].astype(np.float32)
        p = dict_events['p'].astype(np.float32)
        events = np.concatenate([x[:, np.newaxis], y[:, np.newaxis], t[:, np.newaxis], p[:, np.newaxis]], axis=1)
        return torch.from_numpy(events), label


class Loader:
    def __init__(self, dataset, args, device, distributed, batch_size, drop_last=False):
        self.device = device
        if distributed is True:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            split_indices = list(range(len(dataset)))
            self.sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=self.sampler,
                                             num_workers=args.train_num_workers, pin_memory=True,
                                             collate_fn=collate_events, drop_last=drop_last)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = torch.cat([d[0], i*torch.ones((len(d[0]), 1), dtype=torch.float32)], 1)
        events.append(ev)
    events = torch.cat(events, 0)
    labels = default_collate(labels)
    return events, labels
