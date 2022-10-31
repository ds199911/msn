#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=/home/ds5749/jupyter.log

singularity exec --nv \
        --overlay /scratch/projects/shamoutlab/ds5749/MedFuse/medfuse.ext3:ro \
        --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
        --overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
        /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c
        "source /ext3/env.sh; conda activate /ext3/medfuse"

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888



# Launch interactive job with a gpu: srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=32GB --time=4:00:00 --gres=gpu:1 --pty /bin/bash
# Execute singularity container: 


source /ext3/env.sh
conda activate /ext3/medfuse

Start jupyter lab: jupyter notebook --ip 0.0.0.0 --port 8965 --no-browser