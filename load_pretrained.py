import os
import argparse
import logging
import pprint

# import numpy as np
import torch
import torchvision.transforms as transforms
import cyanure as cyan

import src.deit as deit
from src.data_manager import (
    init_data,
)


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_pretrained(
    encoder,
    pretrained
):  
    print(pretrained)
    checkpoint = torch.load(pretrained, map_location='cpu')
    print(checkpoint.keys())
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(msg)
    logger.info(f'loaded pretrained model with msg: {msg}')
    try:
        logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                    f'path: {pretrained}')
    except Exception:
        print(checkpoint.keys())
    del checkpoint
    return encoder


def init_model(
    device,
    pretrained,
    model_name,
):
    encoder = deit.__dict__[model_name]()
    encoder.fc = None
    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained)

    return encoder

device = 'cuda:0'
model_name  = 'deit_small'
pretrained = 'pretrained/vits16_800ep.pth.tar'
print(init_model(
    device,
    pretrained,
    model_name,
))
