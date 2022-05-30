#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:v100l:1
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --job-name=train_demo
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn

source arocr/bin/activate

pwd


python train.py --ENCODER  --DECODER xlm-roberta-base --MODEL_NAME arocr --SEED 42 --DATASET ADAB --DATA_DIR /project/6007993/DataBank/OCR_data/Datasets/al/_Ready/ADAB  
