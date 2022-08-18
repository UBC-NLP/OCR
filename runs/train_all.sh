#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-mageed
#SBATCH --job-name=train_hf_v2
#SBATCH --output=train_hf_v2.out
#SBATCH --error=train_hf_v2.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn


source ~/ocr/bin/activate
#pip3 install datasets
#pip3 install transformers 
#pip3 install torch
#pip3 install wandb

pwd

echo "Training started at $(date)"

python train_all.py

echo "Training ended at $(date)"