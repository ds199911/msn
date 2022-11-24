from __future__ import absolute_import
from __future__ import print_function
import sys 
sys.path.append('..')
import numpy as np
import argparse
import os
import imp
# import re
# from trainers.fusion_trainer import FusionTrainer
# from trainers.mmtm_trainer import MMTMTrainer
# from trainers.daft_trainer import DAFTTrainer

from ehr_utils.preprocessing import Discretizer, Normalizer
from ehr_dataset import get_datasets, get_data_loader
from cxr_dataset import get_cxr_datasets
# from fusion import load_cxr_ehr
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from arguments import args_parser

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
# print(args)

# seed = 1002
# torch.manual_seed(seed)
# np.random.seed(seed)

# def read_timeseries(args):
#     path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
#     ret = []
#     with open(path, "r") as tsfile:
#         header = tsfile.readline().strip().split(',')
#         assert header[0] == "Hours"
#         for line in tsfile:
#             mas = line.strip().split(',')
#             ret.append(np.array(mas))
#     return np.stack(ret)

# discretizer = Discretizer(timestep=float(args.timestep),
#                           store_masks=True,
#                           impute_strategy='previous',
#                           start_time='zero')

# discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
# cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
# normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
# normalizer_state = args.normalizer_state
# if normalizer_state is None:
#     normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
#     normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
# normalizer.load_params(normalizer_state)

# ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
# train_dl, val_dl, test_dl = get_data_loader(discretizer, normalizer, args, args.batch_size)
# cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

from src.data_manager import (
    init_data,
    init_ehr_data,
    make_transforms
)
import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
(unsupervised_loader, unsupervised_sampler) = init_ehr_data(
            batch_size=64,
            pin_mem=True,
            num_workers=0,
            world_size=1,
            rank=0,
            training=True,
            args=args,
            augmentation='vertical_horizontal',
            distributed=False)

start_time = time.time()
for itr, data in enumerate(unsupervised_loader):
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    if itr == 10:
        break
