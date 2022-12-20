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

class LinearClassifier(torch.nn.Module):

    def __init__(self, dim, num_labels=1000, normalize=True):
        super(LinearClassifier, self).__init__()
        self.normalize = normalize
        self.norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        # x = self.norm(x)
        if self.normalize:
            x = torch.nn.functional.normalize(x)
        return self.linear(x)

def load_pretrained(
    r_path,
    encoder,
    linear_classifier,
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
    return encoder, linear_classifier

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
    encoder.fc = ln
    encoder.norm = None

    encoder.to(device)
    encoder, _ = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        linear_classifier=None,
        device_str=device_str)

    linear_classifier = LinearClassifier(512, num_classes, normalize).to(device)

    # -- init optimizer
    optimizer, scheduler = None, None
    param_groups = [
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'weight_decay': 0}
    ]
    optimizer = SGD(
        param_groups,
        nesterov=True,
        weight_decay=weight_decay,
        momentum=0.9,
        lr=ref_lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_epochs*iterations_per_epoch,
        start_lr=ref_lr,
        ref_lr=ref_lr,
        T_max=num_epochs*iterations_per_epoch)
    # if world_size > 1:
    #     linear_classifier = DistributedDataParallel(linear_classifier)

    return encoder, linear_classifier, optimizer, scheduler


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

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    batch_size = 128
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
        args=medfuse_args,
        ehr_augmentation=None,
        distributed=False)

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
        training=False,
        drop_last=False,
        copy_data=copy_data,
        dataset_name='MIMICCXR',
        args=medfuse_args,
        ehr_augmentation=None,
        distributed=False)
    logger.info(f'initialized val data-loader (ipe {len(val_data_loader)})')

     # -- init model and optimizer
    encoder, linear_classifier, optimizer, scheduler = init_model(
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

    best_acc = None
    start_epoch = 0
    logger.info('putting model in eval mode')
    encoder.eval()
    logger.info(sum(p.numel() for n, p in encoder.named_parameters()
                    if p.requires_grad and ('fc' not in n)))
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):

        def train_step():
            # -- update distributed-data-loader epoch
            outGT = torch.FloatTensor().to(device)
            outPRED = torch.FloatTensor().to(device)
            top1_correct, top5_correct, total = 0, 0, 0
            for i, data in enumerate(data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    x, img, labels, seq_length = data[0].to(device).float(), data[1].to(device), torch.tensor(data[2]).to(device).float(), data[4]
                    with torch.no_grad():
                        outputs = encoder(x, seq_length, img, return_before_head=True)
                outputs = linear_classifier(outputs)
                loss = criterion(outputs, labels) 

                outPRED = torch.cat((outPRED, outputs), 0)
                outGT = torch.cat((outGT, labels), 0)

                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                logger.info('epoch: {}'.format(i))
        
            metrics = computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy())
            return metrics

        def val_step():
            outGT = torch.FloatTensor().to(device)
            outPRED = torch.FloatTensor().to(device)
            top1_correct, total = 0, 0
            for i, data in enumerate(val_data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    # inputs, labels = data[0].to(device), data[1].to(device)
                    x, img, labels, seq_length = data[0].to(device).float(), data[1].to(device), torch.tensor(data[2]).to(device).float(), data[4]
                    # outputs = encoder.forward_blocks(inputs, num_blocks)
                    outputs = encoder(x, seq_length, img, return_before_head=True)
                outputs = linear_classifier(outputs)
                outPRED = torch.cat((outPRED, outputs), 0)
                outGT = torch.cat((outGT, labels), 0)
                # total += inputs.shape[0]
                # top1_correct += outputs.max(dim=1).indices.eq(labels).sum()
                # top1_acc = 100. * top1_correct / total

            # top1_acc = AllReduce.apply(top1_acc)
            # logger.info('[%d, %5d] %.3f%%' % (epoch + 1, i, top1_acc))
            # return top1_acc
            metrics = computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy())
            return metrics

        train_top1 = train_step()
        logger.info('train metrics')
        # for key in train_top1:
        #     logger.info(key)
        #     logger.info(train_top1[key])
        logger.info('train auroc_mean:') 
        logger.info(train_top1['auroc_mean'])
        logger.info('train auprc_mean:')
        logger.info(train_top1['auprc_mean'])

        with torch.no_grad():
            val_top1 = val_step()
        logger.info('val metrics')
        logger.info('val auroc_mean: ')
        logger.info(val_top1['auroc_mean'])
        logger.info('val auprc_mean: ')
        logger.info(val_top1['auprc_mean'])
        # for key in val_top1:
        #     logger.info(key)
        #     logger.info(val_top1[key])
        # log_str = 'train:' if training else 'test:'
        # logger.info('[%d] (%s %.3f%%) (val: %.3f%%)'
        #             % (epoch + 1, log_str, train_top1, val_top1))

        # -- logging/checkpointing
        # rank = 0
        # if training and (rank == 0) and ((best_acc is None) or (best_acc < val_top1)):
        #     best_acc = val_top1
        #     save_dict = {
        #         'target_encoder': encoder.state_dict(),
        #         'classifier': linear_classifier.state_dict(),
        #         'opt': optimizer.state_dict(),
        #         'epoch': epoch + 1,
        #         'world_size': 0,
        #         'best_top1_acc': best_acc,
        #         'batch_size': batch_size,
        #         'lr': ref_lr,
        #     }
        #     torch.save(save_dict, w_enc_path)

    return train_top1, val_top1
            
if __name__ == '__main__':
    args = parser.parse_args()
    main()