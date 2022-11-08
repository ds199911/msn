import argparse
import logging
import pprint
import yaml
import sys
import os
sys.path.append('..')

import numpy as np

import torch
import torchvision.transforms as transforms
from src.medfuse.utils import computeAUROC
import src.deit as deit
from src.utils import (
    AllReduce,
    init_distributed,
    WarmupCosineSchedule
)
from src.sgd import SGD

from src.data_manager import (
    init_fusion_data,
    make_transforms
)
from src.medfuse.fusion_models import fusion_model
from src.medfuse.arguments import args_parser

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

parser = args_parser()
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')

def main():

# -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    logger.info('Running linear-evaluation')
    return linear_eval(params, args)

def load_pretrained(
    r_path,
    encoder,
    device_str
):
    checkpoint = torch.load(r_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                f'path: {r_path}')
    del checkpoint
    return encoder

def init_model(
    device,
    device_str,
    num_classes,
    num_blocks,
    training,
    r_enc_path,
    iterations_per_epoch,
    world_size,
    ref_lr,
    num_epochs,
    normalize,
    model_name='resnet50',
    warmup_epochs=0,
    weight_decay=0
):
    # -- init model
    encoder = fusion_model(device)
    # encoder = deit.__dict__[model_name]()
    # emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
    emb_dim = 512
    emb_dim *= num_blocks
    ln = torch.nn.LayerNorm(emb_dim)
    encoder.fc = torch.nn.Linear(emb_dim, num_classes)
    encoder.norm = None

    encoder.to(device)
    encoder = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        device_str=device_str)


    # -- init optimizer
    optimizer, scheduler = None, None
    param_groups = encoder.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=1e-6)
    
    return encoder, optimizer, scheduler


def linear_eval(args, medfuse_args):
    model_name = args['meta']['model_name']
    port = args['meta']['master_port']
    load_checkpoint = args['meta']['load_checkpoint']
    training = args['meta']['training']
    copy_data = args['meta']['copy_data']
    device = torch.device(args['meta']['device'])
    if 'cuda' in args['meta']['device']:
        torch.cuda.set_device(device)

    # -- DATA
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    num_classes = args['data']['num_classes']

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    ref_lr = args['optimization']['lr']
    num_epochs = args['optimization']['epochs']
    num_blocks = args['optimization']['num_blocks']
    l2_normalize = args['optimization']['normalize']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    r_file_enc = args['logging']['pretrain_path']

    # -- log/checkpointing paths
    r_enc_path = os.path.join(folder, r_file_enc)
    w_enc_path = r_file_enc #os.path.join(folder, f'{tag}-lin-eval.pth.tar')
    r_enc_path = r_file_enc

    batch_size = 64
    load_checkpoint = True
    num_epochs = 1

    # -- init loss
    criterion = torch.nn.BCEWithLogitsLoss() #multi-label loss

    # -- make train data transforms and data loaders/samples
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])
    data_loader, dist_sampler = init_fusion_data(
        transform=transform,
        batch_size=batch_size,
        training=training,
        copy_data=copy_data,
        dataset_name='MIMICCXR',
        args=medfuse_args)

    ipe = len(data_loader)
    logger.info(f'initialized data-loader (ipe {ipe})')

    # -- make val data transforms and data loaders/samples
    val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])
    val_data_loader, val_dist_sampler = init_fusion_data(
        transform=val_transform,
        batch_size=batch_size,
        root_path=root_path,
        image_folder=image_folder,
        training=False,
        drop_last=False,
        copy_data=copy_data,
        dataset_name='MIMICCXR',
        args=medfuse_args)
    logger.info(f'initialized val data-loader (ipe {len(val_data_loader)})')

     # -- init model and optimizer
    encoder, optimizer, scheduler = init_model(
        device=device,
        device_str=args['meta']['device'],
        num_classes=num_classes,
        num_blocks=num_blocks,
        normalize=l2_normalize,
        training=training,
        r_enc_path=r_enc_path,
        iterations_per_epoch=ipe,
        world_size=None,
        ref_lr=ref_lr,
        weight_decay=wd,
        num_epochs=num_epochs,
        model_name=model_name)
    logger.info(encoder)

    start_epoch = 0
    logger.info('putting model in training mode')
    encoder.train()
    logger.info(sum(p.numel() for n, p in encoder.named_parameters()
                    if p.requires_grad and ('fc' not in n)))
    start_epoch = 0

    # for epoch in range(start_epoch, num_epochs):

    def train_step(compute_metrics=False):
        # -- update distributed-data-loader epoch
        # dist_sampler.set_epoch(epoch)
        encoder.train()
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        top1_correct, top5_correct, total = 0, 0, 0
        total_loss = 0
        for i, data in enumerate(data_loader):
            with torch.cuda.amp.autocast(enabled=True):
                x, img, labels, seq_length = data[0].to(device).float(), data[1].to(device), torch.tensor(data[2]).to(device).float(), data[4]
                outputs = encoder(x, seq_length, img, return_before_head=True)
            loss = criterion(outputs, labels) 

            outPRED = torch.cat((outPRED, outputs), 0)
            outGT = torch.cat((outGT, labels), 0)
            total_loss += loss.item()
            if training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        logger.info("train loss: {}".format(total_loss/(len(data_loader))))
        if compute_metrics:
            metrics = computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy())
            return metrics
        return None

    def val_step():
        total_loss = 0
        encoder.eval()
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        top1_correct, total = 0, 0
        for i, data in enumerate(val_data_loader):
                x, img, labels, seq_length = data[0].to(device).float(), data[1].to(device), torch.tensor(data[2]).to(device).float(), data[4]
                outputs = encoder(x, seq_length, img, return_before_head=True)
                loss = criterion(outputs, labels) 
                outPRED = torch.cat((outPRED, outputs), 0)
                outGT = torch.cat((outGT, labels), 0)
                total_loss += loss.item()
        logger.info("val loss: {}".format(total_loss/(len(data_loader))))
        metrics = computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy())
        return metrics

    for i in range(1, 50+1):
        logger.info("Epoch:{}".format(i))
        if i%5 == 0:
            train_top1 = train_step(True)
            logger.info('train auroc_mean:') 
            logger.info(train_top1['auroc_mean'])
            logger.info('train auprc_mean:')
            logger.info(train_top1['auprc_mean'])
        else:
            train_top1 = train_step()

        with torch.no_grad():
            val_top1 = val_step()
        logger.info('val auroc_mean: ')
        logger.info(val_top1['auroc_mean'])
        logger.info('val auprc_mean: ')
        logger.info(val_top1['auprc_mean'])

    return train_top1, val_top1
            
if __name__ == '__main__':
    args = parser.parse_args()
    main()