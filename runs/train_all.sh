#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:v100l:4
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=run_all_models_v2
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

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

wandb agent gagan3012/arocr/v5etz5et

echo "Training ended at $(date)"