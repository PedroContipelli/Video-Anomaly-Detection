#!/bin/bash
#SBATCH --nodes=1                      # Get one node
#SBATCH --cpus-per-task=2              # Two cores per task
#SBATCH --ntasks=1                     # But only one task
#SBATCH --gres=gpu:1                   # And one GPU
#SBATCH --gres-flags=enforce-binding   # Insist on good CPU/GPU alignment
#SBATCH --time=0-4:00:00               # Run for 4 hours, at most
#SBATCH --job-name=ad-pedro            # Name the job so I can see it in squeue
#SBATCH --output=./outputs/output-%J.out
#SBATCH --mail-type=END,FAIL     # Send me email for various states
#SBATCH --mail-user pedrocontipelli@knights.ucf.edu        # Use this address

# Load modules
module load anaconda/anaconda3

module list

nvidia-smi topo -m                     # Have Nvidia tell us the GPU/CPU mapping so we know

source activate pytorch-gpu

python3 --version

CUDA_VISIBLE_DEVICES=0
./run.sh
