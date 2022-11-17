from __future__ import absolute_import
from __future__ import print_function

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

# if args.missing_token is not None:
#     from trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer
    
# path = Path(args.save_dir)
# path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)

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
# print('discretizer_header',discretizer_header)
# print([(i, x) for (i, x) in enumerate(discretizer_header)])
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
# print(cont_channels)
normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

# ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
train_dl, val_dl, test_dl = get_data_loader(discretizer, normalizer, args, args.batch_size)

# cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

# print(len(ehr_train_ds[0]))
# print(len(cxr_train_ds[0]))

# train_dl = DataLoader(ehr_train_ds, args.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
# print(train_dl)
# load = iter(train_dl)
# print(load.next())
# for itr, (data, label) in enumerate(train_dl):
#     print(itr)
#     print(data.shape)
#     print(label)
#     break
for itr, batch in enumerate(train_dl):
    # data, label = batch
    a, b, c = batch
    # x, targets, seq_length
    # print(batch)
    # print(len(batch))
    # print(a)
    print(len(a))
    print(a[0].shape)
    print(b)
    print(c)
    # print(b.shape)
    # print(len(c))
    break
    # print(data.shape)
    # print(label)

print(args)