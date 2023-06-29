#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=BP-noev
#SBATCH --error=/home/gs4288/guohao/ELIP/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/ELIP/RC_out/out_%j.txt
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --partition tier3
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=200g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


export MASTER_PORT=12340
export WORLD_SIZE=4

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


source ~/conda/etc/profile.d/conda.sh
conda activate elip-td

srun python -m torch.distributed.run --nproc_per_node=4 /home/gs4288/guohao/ELIP/train_BLIP_retrieval.py --config /home/gs4288/guohao/ELIP/configs/retrieval_blip_noEv.yaml --output_dir /home/gs4288/guohao/ELIP/output/Bp-noEV