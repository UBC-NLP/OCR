#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-mageed
#SBATCH --job-name=predict
#SBATCH --output=predict.out
#SBATCH --error=predict.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

dataset=$1

module load python/3.8 scipy-stack gcc arrow cuda cudnn

source ~/ENV38_default/bin/activate

python predict_v2.py --dataset_name $1