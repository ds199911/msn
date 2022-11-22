# MSN  **M**asked **S**iamese **N**etworks

This repo provides a PyTorch implementation of Multi-Modal MSN

Multi-Modal MSN is a self-supervised learning framework that leverages the idea of mask-denoising while avoiding pixel and token-level reconstruction. Given two views of a multi-modal datapoint, Multi-Modal MSN randomly masks patches from one view of the multi-modal data while leaving the other view unchanged. The objective is to train a neural network encoder, parametrized with a vision transformer (ViT) and LSTM, to output similar embeddings for the two views. In this procedure, Multi-Modal MSN does not predict the masked patches at the input level, but rather performs the denoising step implicitly at the representation level by ensuring that the representation of the masked input matches the representation of the unmasked one.

## Pre-trained models

## Running MSN self-supervised pre-training

### Config files
All experiment parameters are specified in config files (as opposed to command-line-arguments). Config files make it easier to keep track of different experiments, as well as launch batches of jobs at a time. See the [configs/](configs/) directory for example config files.

### Requirements
* Python 3.8 (or newer)
* PyTorch install 1.11.0 (older versions may work too)
* torchvision
* Other dependencies: PyYaml, numpy, opencv, submitit, cyanure

### Single-GPU training
Our implementation starts from the [main.py](main.py), which parses the experiment config file and runs the msn pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run on GPU "0" on a local machine, use the command:
```
python main.py \
  --fname configs/pretrain/msn_vits16.yaml \
  --devices cuda:0
```
<!-- 
### Multi-GPU training
In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster. Feel free to edit [main_distributed.py](main_distributed.py) for your purposes to specify a different procedure for launching a multi-GPU job on a cluster.

For example, to pre-train with MSN on 16 GPUs using the pre-training experiment configs specificed inside [configs/pretrain/msn_vits16.yaml](configs/pretrain/msn_vits16.yaml), run:
```
python main_distributed.py \
  --fname configs/pretrain/msn_vits16.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
``` -->

## Linear Evaluation
To run linear evaluation on the entire Medfuse Dataset, use the `linear_eval.py` script.

For example, for ehr data run:
```
python mefuse_linear_eval.py --fname configs/eval/medfuse_ehr.yaml --modality ehr
```

cxr data run:
```
python mefuse_linear_eval.py --fname configs/eval/medfuse_cxr_vits16.yaml
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
