# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
from collections import OrderedDict

import numpy as np

import torch
import torch.multiprocessing as mp

import src.deit as deit
from src.utils import (
    AllReduceSum,
    trunc_normal_,
    gpu_timer,
    init_distributed,
    WarmupCosineSchedule,
    CosineWDSchedule,
    CSVLogger,
    grad_logger,
    AverageMeter
)
from src.losses import init_msn_loss
from src.data_manager import (
    init_data,
    make_transforms
)

from torch.nn.parallel import DistributedDataParallel

# --
log_timings = True
log_freq = 10
checkpoint_freq = 25
checkpoint_freq_itr = 2500
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    model_name = args['meta']['model_name']
    two_layer = False if 'two_layer' not in args['meta'] else args['meta']['two_layer']
    bottleneck = 1 if 'bottleneck' not in args['meta'] else args['meta']['bottleneck']
    output_dim = args['meta']['output_dim']
    hidden_dim = args['meta']['hidden_dim']
    load_model = args['meta']['load_checkpoint']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    use_pred_head = args['meta']['use_pred_head']
    use_bn = args['meta']['use_bn']
    drop_path_rate = args['meta']['drop_path_rate']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- CRITERTION
    memax_weight = 1 if 'memax_weight' not in args['criterion'] else args['criterion']['memax_weight']
    ent_weight = 1 if 'ent_weight' not in args['criterion'] else args['criterion']['ent_weight']
    freeze_proto = False if 'freeze_proto' not in args['criterion'] else args['criterion']['freeze_proto']
    use_ent = False if 'use_ent' not in args['criterion'] else args['criterion']['use_ent']
    reg = args['criterion']['me_max']
    use_sinkhorn = args['criterion']['use_sinkhorn']
    num_proto = args['criterion']['num_proto']
    # --
    batch_size = args['criterion']['batch_size']
    temperature = args['criterion']['temperature']
    _start_T = args['criterion']['start_sharpen']
    _final_T = args['criterion']['final_sharpen']

    # -- DATA
    label_smoothing = args['data']['label_smoothing']
    pin_mem = False if 'pin_mem' not in args['data'] else args['data']['pin_mem']
    num_workers = 1 if 'num_workers' not in args['data'] else args['data']['num_workers']
    color_jitter = args['data']['color_jitter_strength']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    patch_drop = args['data']['patch_drop']
    rand_size = args['data']['rand_size']
    rand_views = args['data']['rand_views']
    focal_views = args['data']['focal_views']
    focal_size = args['data']['focal_size']
    # --

    # -- OPTIMIZATION
    clip_grad = args['optimization']['clip_grad']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    # ----------------------------------------------------------------------- #

    # try:
    #     mp.set_start_method('spawn')
    #     print('multiprocess')
    # except Exception:
    #     pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- proto details
    assert num_proto > 0, 'unsupervised pre-training requires specifying prototypes'

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
 

    # -- make csv_logger
    # -- init model
    # -- init losses
    # -- make data transforms
    print('focal_views', focal_views)
    print('rand_views', rand_views+1)
    transform = make_transforms(
        rand_size=rand_size,
        focal_size=focal_size,
        rand_views=rand_views+1,
        focal_views=focal_views,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         transform=transform,
         batch_size=batch_size,
         pin_mem=pin_mem,
         num_workers=1,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=True,
         copy_data=copy_data)
    ipe = len(unsupervised_loader)
    logger.info(f'iterations per epoch: {ipe}')

    # -- make prototypes
    # -- init optimizer and scheduler
    # -- momentum schedule

    start_epoch = 0
    # -- load training checkpoint
    # if load_model:
    #     encoder, target_encoder, prototypes, saved_optimizer, start_epoch = load_checkpoint(
    #         device=device,
    #         prototypes=prototypes,
    #         r_path=load_path,
    #         encoder=encoder,
    #         target_encoder=target_encoder,
    #         opt=optimizer)
    #     if saved_optimizer is not None:
    #         optimizer = saved_optimizer
    #     for _ in range(start_epoch*ipe):
    #         scheduler.step()
    #         wd_scheduler.step()
    #         next(momentum_scheduler)
    #         next(sharpen_scheduler)
    

    # -- TRAINING LOOP
    for epoch in range(start_epoch, 1):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        # unsupervised_sampler.set_epoch(epoch)
        for itr, (udata, y) in enumerate(unsupervised_loader):
            logger.info('start iter')
            def load_imgs():
                # -- unsupervised imgs
                imgs = [u.to(device, non_blocking=True) for u in udata]
                return imgs
            imgs = load_imgs()
            print('y:',len(y), type(y))
            print('img:',len(imgs))
            print('anchor:',len(imgs[1:]))
            print('target:',len(imgs[0]))
            # imgs, dtime = gpu_timer(load_imgs)
            # data_meter.update(dtime)   
            print(len(udata))
            break
        break
            

        
if __name__ == "__main__":
    import yaml
    import pprint

    fname = 'configs/pretrain/msn_vits16.yaml'
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
    dump = os.path.join(params['logging']['folder'], 'params-msn-train.yaml')
    with open(dump, 'w') as f:
        yaml.dump(params, f)
    main(params)
