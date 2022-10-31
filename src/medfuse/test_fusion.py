
import numpy as np
import os

from ehr_utils.preprocessing import Discretizer, Normalizer

import torch
from torch.utils.data import DataLoader
from arguments import args_parser

from fusion_dataset import load_cxr_ehr
from ehr_dataset import get_datasets
from cxr_dataset import get_cxr_datasets

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
print(args)

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
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)
args.data_pairs = 'partial_ehr_cxr'
args.fusion_type = 'lstm'
print('load datasets')
ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
print('loaded ehr')
cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets()

train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)
print('loaded dataloaders')
for itr, batch in enumerate(train_dl):
    # data, label = batch
    x, img, targets_ehr, targets_cxr, seq_length, pairs = batch
    # x, targets, seq_length
    # print(batch)
    # print(len(batch))
    try:
        print('x.shape')
        print(x.shape)
    except:
        print(len(x))
        print(len(x[0]))
        print(x[0][0].shape)
    print(len(img))
    print(len(img[0]))
    print(img[0][0].shape)
    # print(img)
    print(targets_ehr.shape)
    # print(targets_cxr)
    # print(pairs)

    # print(b.shape)
    # print(len(c))
    break
    # print(data.shape)
    # print(label)

print(args)