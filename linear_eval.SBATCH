#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
module purge

# singularity exec --nv \
#         --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
#         --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
#         --overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
#         /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
#         "source /ext3/env.sh; conda activate /ext3/medfuse;
#        python main_distributed.py \
#         --linear-eval \
#         --fname configs/eval/lineval_msn_vits16.yaml \
#         --folder checkpoint/eval \
#         --partition 4 \
#         --nodes 4 --tasks-per-node 8 \
#         --time 1000"

# singularity exec --nv \
        # --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        # --overlay /scratch/projects/shamoutlab/data/data.sqf:ro \
        # /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
        # "source /ext3/env.sh; conda activate /ext3/medfuse;
        # python mefuse_linear_eval.py --fname configs/eval/medfuse_cxr_vits16.yaml"

singularity exec --nv \
        --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        --overlay /scratch/projects/shamoutlab/data/data.sqf:ro \
        /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
        "source /ext3/env.sh; conda activate /ext3/medfuse;
        python mefuse_linear_eval.py --fname configs/eval/medfuse_ehr.yaml --modality ehr"