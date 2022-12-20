import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from logging import getLogger
from random import choice
logger = getLogger()
class EHRdataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, dataset_dir, return_names=True,  transforms=None, period_length=48.0,):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length
        self.transforms = transforms
        self._dataset_dir = dataset_dir
        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]


        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[0]: {
                'labels': list(map(float, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }

        self.names = list(self.data_map.keys())
    
    def _read_timeseries(self, ts_filename, time_bound=None):
        
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def read_by_file_name(self, index, time_bound=None):
        t = self.data_map[index]['time'] if time_bound is None else time_bound
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        (X, header) = self._read_timeseries(index, time_bound=time_bound)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}

    def get_decomp_los(self, index, time_bound=None):
        return self.__getitem__(index, time_bound)


    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self._period_length

        if self.transforms is not None:
            data = self.transforms(data)
            for i in range(len(data)):
                data[i] = self.discretizer.transform(data[i], end=ts)[0] 
                if 'gaussian' in self.transforms.augmentation and i != 0:
                    data[i] = self.transforms.gaussian_blur(data[i])
                if 'rotation' in self.transforms.augmentation and i != 0:
                    data[i] = self.transforms.rotation(data[i])
                if 'sampling' in self.transforms.augmentation and i != 0:
                    data[i] = self.transforms.downsample(data[i])
                if (self.normalizer is not None):
                    data[i] = self.normalizer.transform(data[i])
        else:
            data = self.discretizer.transform(data, end=ts)[0] 
            if (self.normalizer is not None):
                data = self.normalizer.transform(data)
        ys = ret["y"]
        # names = ret["name"]
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, ys
    
    def __len__(self):
        return len(self.names)


def get_datasets(discretizer, normalizer, args, augmentation=None):
    if augmentation:
        cur_len = 0
        begin_pos, categorical = [0]*len(discretizer._id_to_channel), []
        for i in range(len(discretizer._id_to_channel)-1):
            begin_pos[i+1] = begin_pos[i] + max(len(discretizer._possible_values[discretizer._id_to_channel[i]]), 1)
            if discretizer._is_categorical_channel[discretizer._id_to_channel[i]]:
                categorical.append(i)
        [begin_pos.pop(categorical[i]-i) for i in range(len(categorical))]

        transform = MultiTransform(views=11, normal_values=discretizer._id_normal_values, _is_categorical_channel=discretizer._is_categorical_channel, augmentation=augmentation, begin_pos=begin_pos)
        train_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'), transforms=transform)
    else: 
        train_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))

    val_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
    test_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/test'))
    return train_ds, val_ds, test_ds

def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl, test_dl
        

# def my_collate(batch):
#     x = [item[0] for item in batch]
#     x, seq_length = pad_zeros(x)
#     targets = np.array([item[1] for item in batch])
#     return [x, targets, seq_length]

def my_collate(batch):
    x = [item[0] for item in batch]
    targets = np.array([item[1] for item in batch])
    if isinstance(x[0], list):
        x, seq_length = pad_zeros_mask(x)
    else:
        x, seq_length = pad_zeros(x)
        x = torch.from_numpy(x)
    return [x, targets, seq_length]


def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length

def pad_zeros_mask(arr, min_length=None):
    seq_length = [x[0].shape[0] for x in arr]
    max_len = max(seq_length)
    ret = []
    for xs in arr:
        ret.append([torch.cat([torch.from_numpy(x), torch.zeros((max_len - x.shape[0],) + x.shape[1:])], axis=0)
        for x in xs])  
    if (min_length is not None) and ret[0].shape[0] < min_length:
        for xs in arr:
            ret.append([torch.cat([torch.from_numpy(x), np.zeros((min_length - x.shape[0],) + x.shape[1:])], axis=0)
            for x in xs]) 
    res = []
    for i in range(len(ret[0])):
        batch = []
        for j in range(len(ret)):
            batch.append(ret[j][i])
        res.append(torch.stack(batch))
    return res, seq_length


class MultiTransform(object):

    def __init__(
        self,
        views,
        normal_values,
        _is_categorical_channel,
        augmentation,
        begin_pos
    ):
        self.views = views
        self.normal_values = normal_values
        self.rows = np.array([value for value in self.normal_values.values()])
        self.augmentation = augmentation
        self.continuous_variable = [0 if _is_categorical_channel[key] == True else 1 for key in _is_categorical_channel]
        self.begin_pos = begin_pos
    def vertical_mask(self, data, max_percent=0.4):
        # mask over each timestep (t, features)
        length = data.shape[0]
        if length < 4:
            return data
        size = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        a = np.zeros(length , dtype=int)
        a[:size] = 1
        np.random.shuffle(a)
        a = a.astype(bool)
        data[a,1:] = self.rows
        return data

    def horizontal_mask(self, data, max_percent=0.4):
        # mask over each feature (t, features)
        length = data.shape[1] - 1
        size = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        features = np.unique(np.random.randint(low=1, high=length, size=size))
        for i in features:
            data[:,i+1] = self.normal_values[i]
        return data
    
    def drop_start(self, data, max_percent=0.4):
        length = data.shape[0]
        start = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        return data[start:,:]

    def gaussian_blur(self, data):
        mean, std = 1,0 
        data[:, self.begin_pos] = data[:, self.begin_pos]  + np.random.normal(mean, std, (data.shape[0], len(self.begin_pos)))
        return data

    def rotation(self, data):
        if choice([0,1]):
            return np.flip(data, axis=0)
        return data

    def downsample(self, data):
        if data.shape[0] < 20:
            return data
        step = choice([1, 2, 3])
        return data[::step]

    def __call__(self, img):
        img_views = [img]

        # -- generate random views
        if self.views > 0:
            if self.augmentation == 'vertical_horizontal':
                for _ in range(self.views//2):
                    img_views.append(self.vertical_mask(img))
                for _ in range(self.views - (self.views//2)):
                    img_views.append(self.horizontal_mask(img))
            elif self.augmentation == 'vertical_and_horizontal':
                for _ in range(self.views):
                    img_views.append(self.horizontal_mask(self.vertical_mask(img)))
            elif self.augmentation == 'drop_start':
                for _ in range(self.views):
                    img_views.append((self.drop_start(img)))
            else:
                for _ in range(self.views):
                    img_views.append(img)

        return img_views