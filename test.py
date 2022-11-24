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

# from ehr_utils.preprocessing import Discretizer, Normalizer
# from ehr_dataset import get_datasets, get_data_loader
# from cxr_dataset import get_cxr_datasets
# from fusion import load_cxr_ehr
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.medfuse.arguments import args_parser

parser = args_parser()
args = parser.parse_args()
# print(args)

from src.data_manager import (
    init_data,
    init_ehr_data,
    make_transforms
)
import time
start_time = time.time()
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
# 'vertical_horizontal'
start_time = time.time()
for itr, data in enumerate(unsupervised_loader):
    if itr > 3:
        print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    # print(data[0][0].shape)
    if itr == 13:
        break