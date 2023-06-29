#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=CLIP-flk-idT
#SBATCH --error=/home/gs4288/guohao/ELIP/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/ELIP/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00-1:00:00
#SBATCH --partition tier3
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=60g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate elip-td

srun python -m torch.distributed.run --nproc_per_node=1 /home/gs4288/guohao/ELIP/train_retrieval.py --config /home/gs4288/guohao/ELIP/configs/retrieval_flickr_elip.yaml --output_dir /home/gs4288/guohao/ELIP/output/ELIP-fk-eval_idt --evaluate