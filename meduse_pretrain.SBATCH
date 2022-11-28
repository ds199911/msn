#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH -t 40:00:00
#SBATCH --gres=gpu
#SBATCH --mem=32GB
#SBATCH --job-name=ehr_pretrain-128-vertical_and_horizontal
module purge

singularity exec --nv \
        --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        --overlay /vast/ds5749/data.sqf:ro \
        /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c \
         "source /ext3/env.sh; conda activate /ext3/medfuse;
        python main.py \
         --fname configs/pretrain/msn_fusion_random.yaml \
         --devices cuda:0 \
         --modality fusion"

        # "source /ext3/env.sh; conda activate /ext3/medfuse;
        # python main.py \
        #  --fname configs/pretrain/msn_ehr_lstm.yaml \
        #  --modality ehr \
        #  --devices cuda:0"


        #  python mefuse_linear_eval.py --fname configs/eval/medfuse_cxr_vits16.yaml

         python main.py \
         --fname configs/pretrain/msn_cxr_random.yaml \
         --devices cuda:0

        #  "source /ext3/env.sh; conda activate /ext3/medfuse;
        # python main.py \
        #  --fname configs/pretrain/msn_fusion_random.yaml \
        #  --devices cuda:0 \
        #  --modality fusion" 
        # :v100:1

        # python main.py --fname configs/pretrain/msn_ehr_lstm.yaml --devices cuda:0 --modality ehr #SBATCH --gres=gpu