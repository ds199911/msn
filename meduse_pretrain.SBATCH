#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=medfuse_fusion_pretrain-512-random

module purge

singularity exec --nv \
        --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        --overlay /scratch/projects/shamoutlab/data/data.sqf:ro \
        /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
        "source /ext3/env.sh; conda activate /ext3/medfuse;
        python main.py \
         --fname configs/pretrain/msn_fusion_random.yaml \
         --devices cuda:0 \
         --modality fusion"


        #  python mefuse_linear_eval.py --fname configs/eval/medfuse_cxr_vits16.yaml

        #  python main.py \
        #  --fname configs/pretrain/msn_cxr_random.yaml \
        #  --devices cuda:0