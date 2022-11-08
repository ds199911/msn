#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
module purge


singularity exec --nv \
        --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        --overlay /scratch/projects/shamoutlab/data/data.sqf:ro \
        /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
        "source /ext3/env.sh; conda activate /ext3/medfuse;
        python fusion_linear_eval.py --fname /scratch/projects/shamoutlab/ds5749/multi-modal-msn/configs/eval/medfuse_fusion.yaml"