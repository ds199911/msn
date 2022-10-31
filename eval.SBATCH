#!/bin/bash
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
module purge

singularity exec --nv \
        --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
        --overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
        /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
        "source /ext3/env.sh; conda activate /ext3/medfuse;
       python logistic_eval.py \
        --subset-path imagenet_subsets1/5imgs_class.txt \
        --root-path .  --image-folder /imagenet/ \
        --device cuda:0 \
        --pretrained pretrained \
        --fname vits16_800ep.pth.tar \
        --model-name deit_small \
        --penalty l2 \
        --lambd 0.0025"