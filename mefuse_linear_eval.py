import argparse
import logging
import pprint
import yaml
import sys
import os

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
from src.data_manager import init_data, init_ehr_data
from src.sgd import SGD
from src.medfuse.arguments import args_parser
from src.medfuse.ehr_models import LSTM
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = args_parser()
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--modality', type=str,
    default='img')
def main():

# -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # dump = params['logging']['folder'] +  f'params-train.yaml'
    # with open(dump, 'w') as f:
    #     yaml.dump(params, f)
    logger.info('Running linear-evaluation')
    if args.modality == 'ehr':
        return linear_eval(params, modality=args.modality, medfuse_params=args)
    return linear_eval(params, modality=args.modality)

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
        # x = self.norm(x.float())
        # if self.normalize:
        #     x = torch.nn.functional.normalize(x)
        return self.linear(x.float())

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

# def load_from_path(
#     r_path,
#     encoder,
#     linear_classifier,
#     opt,
#     sched,
#     device_str
# ):
#     encoder, linear_classifier = load_pretrained(r_path, encoder, linear_classifier, device_str)
#     checkpoint = torch.load(r_path, map_location=device_str)

#     best_acc = None
#     if 'best_top1_acc' in checkpoint:
#         best_acc = checkpoint['best_top1_acc']

#     epoch = checkpoint['epoch']
#     logger.info(f'read-path: {r_path}')
#     del checkpoint
#     # return encoder, opt, sched, epoch, best_acc
#     return encoder, linear_classifier, None, None, None, best_acc

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
    weight_decay=0,
    modality='img'
):
    # -- init model
    if modality == 'img':
        encoder = deit.__dict__[model_name]()
        emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
        emb_dim *= num_blocks
    elif modality == 'ehr':
        encoder = LSTM()
        emb_dim = 128
    encoder.fc = None
    encoder.norm = None

    encoder.to(device)
    encoder, _ = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        linear_classifier=None,
        device_str=device_str)

    linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)

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


def linear_eval(args, modality='img', medfuse_params=None):
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
    # criterion = torch.nn.CrossEntropyLoss()

    if medfuse_params is None:
        # -- make train data transforms and data loaders/samples
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        data_loader, dist_sampler = init_data(
            transform=transform,
            batch_size=batch_size,
            world_size=None,
            rank=None,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=copy_data)
        
        # -- make val data transforms and data loaders/samples
        val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])
        val_data_loader, val_dist_sampler = init_data(
            transform=val_transform,
            batch_size=batch_size,
            world_size=None,
            rank=None,
            root_path=root_path,
            image_folder=image_folder,
            training=False,
            drop_last=False,
            copy_data=copy_data)
    elif modality == 'ehr':
        data_loader, dist_sampler = init_ehr_data(
            batch_size=batch_size,
            world_size=None,
            rank=None,
            training=training,
            args=medfuse_params,
            augmentation=False,
            distributed=False)    
        val_data_loader, val_dist_sampler = init_ehr_data(
            batch_size=batch_size,
            world_size=None,
            rank=None,
            training=False,
            args=medfuse_params,
            augmentation=False,
            distributed=False) 
    ipe = len(data_loader)
    logger.info(f'initialized data-loader (ipe {ipe})')
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
        model_name=model_name,
        modality=modality)
    logger.info(encoder)

    best_acc = None
    start_epoch = 0
    # -- load checkpoint
    # encoder, linear_classifier, _, _, _, _ = load_from_path(
    #     r_path=w_enc_path,
    #     encoder=encoder,
    #     linear_classifier=linear_classifier,
    #     opt=optimizer,
    #     sched=scheduler,
    #     device_str=args['meta']['device'])
    logger.info('putting model in eval mode')
    encoder.eval()
    logger.info(sum(p.numel() for n, p in encoder.named_parameters()
                    if p.requires_grad and ('fc' not in n)))
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):

        def train_step():
            # -- update distributed-data-loader epoch
            # dist_sampler.set_epoch(epoch)
            outGT = torch.FloatTensor().to(device)
            outPRED = torch.FloatTensor().to(device)
            for i, data in enumerate(data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    if modality == 'ehr':
                        inputs, labels = data[0].to(device), torch.from_numpy(data[1]).to(device).float()
                    else:
                        inputs, labels = data[0].to(device), data[1].to(device)
                    with torch.no_grad():
                        if modality == 'ehr':
                            outputs = encoder.forward_features_ehr(inputs.half(), data[2])
                        else:
                            outputs = encoder.forward_blocks(inputs, num_blocks)

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
            for i, data in enumerate(val_data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    inputs, labels = torch.tensor(data[0]).to(device), torch.tensor(data[1]).to(device)
                    if modality == 'ehr':
                        outputs = encoder.forward_features_ehr(inputs.half(), data[2])
                    else:
                        outputs = encoder.forward_blocks(inputs, num_blocks)

                outputs = linear_classifier(outputs)
                outPRED = torch.cat((outPRED, outputs), 0)
                outGT = torch.cat((outGT, labels), 0)
                logger.info('val epoch: {}'.format(i))
            metrics = computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy())
            return metrics

        train_metrics = train_step()
        logger.info('train metrics')
        for key in train_metrics:
            logger.info(key)
            logger.info(train_metrics[key])
        with torch.no_grad():
            val_metrics = val_step()
        logger.info('val metrics')
        for key in val_metrics:
            logger.info(key)
            logger.info(val_metrics[key])
        # log_str = 'train:' if training else 'test:'
        # logger.info('[%d] (%s %.3f%%) (val: %.3f%%)'
        #             % (epoch + 1, log_str, train_metrics, val_metrics))

        # -- logging/checkpointing
        # rank = 0
        # if training and (rank == 0) and ((best_acc is None) or (best_acc < val_metrics)):
        #     best_acc = val_metrics
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

    return train_metrics, val_metrics
            
if __name__ == '__main__':
    args = parser.parse_args()
    main()