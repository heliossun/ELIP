#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=Blip-eval-idt
#SBATCH --error=/home/gs4288/guohao/ELIP/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/ELIP/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0-15:00:00
#SBATCH --partition tier3
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=200g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate elip-td

srun python -m torch.distributed.run --nproc_per_node=2 /home/gs4288/guohao/ELIP/train_BLIP_retrieval.py --config /home/gs4288/guohao/ELIP/configs/blip_retrieval-eval.yaml --output_dir /home/gs4288/guohao/ELIP/output/BLIP-eval-idt --evaluate