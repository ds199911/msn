import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from PIL import ImageFilter

import torch
import torchvision

class MIMICCXR(Dataset):
    def __init__(self, split='train', transform=None, path = None, args=None):
        self.data_dir ='/data/MedFuse/2.0.0' #args.cxr_data_dir
        paths = glob.glob(f'{self.data_dir}/resized/**/*.jpg', recursive = True)
        # self.args = args
        self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
        self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

        metadata = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-metadata.csv')
        labels = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv')
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)
        
        splits = pd.read_csv(f'{self.data_dir}/mimic-cxr-ehr-split.csv')


        metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')


        self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
        self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
        self.transform = transform
        self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]

    def __getitem__(self, index):
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert('RGB')
            labels = torch.tensor(self.filesnames_to_labels[index]).float()

            if self.transform is not None:
                img = self.transform(img)
            return img, labels
        
        filename = self.filenames_loaded[index]
        
        img = Image.open(self.filenames_to_path[filename]).convert('RGB')

        labels = torch.tensor(self.filesnames_to_labels[filename]).float()
       
        # labels = torch.argmax(labels)

        if self.transform is not None:
            img = self.transform(img)
        return img, labels
    
    def __len__(self):
        return len(self.filenames_loaded)



def get_cxr_datasets(transform):
    # data_dir = args.cxr_data_dir
    data_dir = '/data/MedFuse/2.0.0'

    paths = glob.glob(f'{data_dir}/resized/**/*.jpg', recursive = True)
    # transform = make_transforms(
    #     rand_size=224,
    #     focal_size=96,
    #     rand_views=1+1,
    #     focal_views=10,
    #     color_jitter=0.5)

    dataset_train = MIMICCXR(split='train', transform=transform)
    dataset_validate = MIMICCXR(split='validate')
    dataset_test = MIMICCXR(split='test')

    return dataset_train, dataset_validate, dataset_test


def make_transforms(
    rand_size=224,
    focal_size=96,
    rand_crop_scale=(0.3, 1.0),
    focal_crop_scale=(0.05, 0.3),
    color_jitter=1.0,
    rand_views=2,
    focal_views=10,
):

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=color_jitter),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])
    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=color_jitter),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=rand_views,
        focal_views=focal_views
    )
    return transform


class MultiViewTransform(object):

    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,
        focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- generate random views
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

