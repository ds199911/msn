# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import dill as pickle
import os
import subprocess
import time

from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()

from src.medfuse.cxr_dataset import MIMICCXR
from src.medfuse.fusion_dataset import load_cxr_ehr
from src.medfuse.ehr_dataset import get_datasets
from src.medfuse.cxr_dataset import get_cxr_datasets
from src.medfuse.ehr_utils.preprocessing import Discretizer, Normalizer
import numpy as np


def init_data(
    transform,
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    dataset_name='MIMICCXR',
    distributed=True
):
    if dataset_name == 'MIMICCXR':
        if training:
            dataset = MIMICCXR(split='train', transform=transform)
            logger.info('MIMICCXR dataset created')
        else:
            dataset = MIMICCXR(split='validate', transform=transform)
            logger.info('MIMICCXR dataset created')
    else:
        dataset = ImageNet(
            root=root_path,
            image_folder=image_folder,
            transform=transform,
            train=training,
            copy_data=copy_data)
        if subset_file is not None:
            dataset = ImageNetSubset(dataset, subset_file)
        logger.info('ImageNet dataset created')
    if distributed:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers)
        dist_sampler=None

    logger.info('unsupervised data loader created')

    return (data_loader, dist_sampler)

def init_fusion_data(
    transform,
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    dataset_name='MIMICCXR',
    args=None,
    rand_views=2,
    distributed=True
):
    if dataset_name == 'MIMICCXR':
        logger.info('MIMICCXR cxr_ehr fusion dataset')

        args.data_pairs = 'partial_ehr_cxr'
        args.fusion_type = 'lstm'
        logger.info(args)
        def read_timeseries(args):
            path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
            ret = []
            with open(path, "r") as tsfile:
                header = tsfile.readline().strip().split(',')
                assert header[0] == "Hours"
                for line in tsfile:
                    mas = line.strip().split(',')
                    ret.append(np.array(mas))
            return np.stack(ret)

        discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
        discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)

        logger.info(args.data_pairs + args.fusion_type)

        ehr_train_ds, ehr_val_ds, cxr_test_ds = get_datasets(discretizer, normalizer, args, augmentation=True)
        cxr_train_ds, cxr_val_ds, ehr_test_ds = get_cxr_datasets(transform)
        train_ds, val_ds, _ = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)
        logger.info('MIMICCXR dataset created')
        
        if training: dataset = train_ds
        else: dataset = val_ds
        
        def my_collate(batch):
            if isinstance(batch[0][1], list):
                img = []
                for i in range(len(batch[0][0])):
                    imgs = []
                    for j in range(len(batch)):
                        try:
                            imgs.append(batch[j][1][i])
                        except:
                            if i < rand_views: 
                                imgs.append(torch.zeros(3, 224, 224)) #rand_size: 224
                            else:
                                imgs.append(torch.zeros(3, 96, 96)) # focal_size: 96
                    img.append(torch.stack(imgs))
            else:
                img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])

            x = [item[0] for item in batch]
            if isinstance(x[0], list):
                x, _ = pad_zeros_mask(x)
                seq_length = [item[0][0].shape[0] for item in batch]
            else:
                x, seq_length = pad_zeros(x)
            targets_ehr = np.array([item[2] for item in batch])
            targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
            pairs = [False if item[1] is None else True for item in batch]
            return [x, img, targets_ehr, targets_cxr, seq_length, pairs]
        
    if distributed:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=0,
            collate_fn=my_collate)
        dist_sampler=None

    logger.info('unsupervised data loader created')

    return (data_loader, dist_sampler)


def init_ehr_data(
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    dataset_name='MIMICCXR',
    args=None,
    augmentation='vertical_horizontal',
    distributed=False
):
    if dataset_name == 'MIMICCXR':
        logger.info('MIMICCXR ehr fusion dataset')

        args.data_pairs = 'partial_ehr_cxr'
        args.fusion_type = 'lstm'
        logger.info(args)
        def read_timeseries(args):
            path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
            ret = []
            with open(path, "r") as tsfile:
                header = tsfile.readline().strip().split(',')
                assert header[0] == "Hours"
                for line in tsfile:
                    mas = line.strip().split(',')
                    ret.append(np.array(mas))
            return np.stack(ret)

        discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
        discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)

        logger.info(args.data_pairs + args.fusion_type)

        ehr_train_ds, ehr_val_ds, cxr_test_ds = get_datasets(discretizer, normalizer, args, augmentation=augmentation)
        logger.info('MIMICCXR dataset created')
        
        if training: dataset = ehr_train_ds
        else: dataset = ehr_val_ds
      
        def my_collate(batch):
            x = [item[0] for item in batch]
            targets = np.array([item[1] for item in batch])
            if isinstance(x[0], list):
                x, seq_length = pad_zeros_mask(x)
            else:
                x, seq_length = pad_zeros(x)
            return [x, targets, seq_length]
        if distributed:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=dist_sampler,
                batch_size=batch_size,
                drop_last=drop_last,
                pin_memory=pin_mem,
                num_workers=0,
                collate_fn=my_collate)
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                pin_memory=pin_mem,
                num_workers=0,
                collate_fn=my_collate)
            dist_sampler=None

    logger.info('unsupervised data loader created')

    return (data_loader, dist_sampler)

def pad_zeros(arr, min_length=None):
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [torch.cat([torch.tensor(x), torch.zeros((max_len - x.shape[0],) + x.shape[1:])], axis=0)
        for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [torch.cat([x, np.zeros((min_length - x.shape[0],) + x.shape[1:])], axis=0)
            for x in ret]
    return torch.stack(ret), seq_length

def pad_zeros_mask(arr, min_length=None):
    max_len = max([x[0].shape[0] for x in arr])
    seq_length = [x[0].shape[0] for x in arr]
    ret = []
    for xs in arr:
        ret.append([torch.cat([torch.tensor(x), torch.zeros((max_len - x.shape[0],) + x.shape[1:])], axis=0)
        for x in xs])  
    if (min_length is not None) and ret[0].shape[0] < min_length:
        for xs in arr:
            ret.append([torch.cat([torch.tensor(x), np.zeros((min_length - x.shape[0],) + x.shape[1:])], axis=0)
            for x in xs]) 
    res = []
    for i in range(len(ret[0])):
        batch = []
        for j in range(len(ret)):
            batch.append(ret[j][i])
        res.append(torch.stack(batch))
    return res, seq_length

def make_transforms(
    rand_size=224,
    focal_size=96,
    rand_crop_scale=(0.3, 1.0),
    focal_crop_scale=(0.05, 0.3),
    color_jitter=1.0,
    rand_views=2,
    focal_views=10,
):
    logger.info('making data transforms')

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


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_folder='imagenet_full_size/',
        tar_file='imagenet_full_size-061417.tar',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_folder=tar_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


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


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_folder='imagenet_full_size/',
    tar_file='imagenet_full_size-061417.tar',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_folder, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path
