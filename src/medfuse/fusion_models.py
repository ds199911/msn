import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../..')
import src.deit as deit
from src.medfuse.ehr_models import LSTM
from collections import OrderedDict

class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
	
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        # target_classes = self.args.num_classes
        self.ehr_dim = 128
        self.cxr_dim = 256
        self.joint_dim = 384
        self.projection = nn.Linear(self.ehr_dim+self.cxr_dim, self.joint_dim)

    def forward(self, x, seq_lengths=None, img=None, return_before_head=True,patch_drop=0., pairs=None):
        ehr_feature, _ = self.ehr_model(x, seq_lengths)   
        cxr_feature = self.cxr_model(img, return_before_head=True, patch_drop=patch_drop)
        input = torch.cat([ehr_feature, cxr_feature], dim=1)
        joint_feature = self.projection(input)
        return joint_feature

def fusion_model(
    device,
    model_name='resnet50',
    use_pred=False,
    use_bn=False,
    two_layer=False,
    bottleneck=1,
    hidden_dim=2048,
    output_dim=128,
    drop_path_rate=0.1,
    ):

    #cxr image model 
    model_name = 'deit_small'
    use_bn = True
    output_dim = 256
    drop_path_rate = 0.0

    cxr_encoder = deit.__dict__[model_name](drop_path_rate=drop_path_rate)
    # emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
    # fc = OrderedDict([])
    # fc['fc1'] = torch.nn.Linear(emb_dim, hidden_dim)
    # if use_bn:
    #     fc['bn1'] = torch.nn.BatchNorm1d(hidden_dim)
    # fc['gelu1'] = torch.nn.GELU()
    # fc['fc2'] = torch.nn.Linear(hidden_dim, hidden_dim)
    # if use_bn:
    #     fc['bn2'] = torch.nn.BatchNorm1d(hidden_dim)
    # fc['gelu2'] = torch.nn.GELU()
    # fc['fc3'] = torch.nn.Linear(hidden_dim, output_dim)
    # cxr_encoder.fc = torch.nn.Sequential(fc)
    
    
    # ehr time series model
    ehr_encoder = LSTM()

    fusion_model = Fusion(args=None, ehr_model=ehr_encoder, cxr_model=cxr_encoder)
    fusion_model.to(device) 
    return fusion_model

