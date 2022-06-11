#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100l:4
#SBATCH --account=def-mageed
#SBATCH --mail-user=ghaniahsan@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=adab
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn


source ~/ocr/bin/activate
pip3 install datasets
pip3 install transformers
pip3 install torch
pip3 install wandb

pwd

echo "Training started at $(date)"

python train.py \
    --model_name_or_path arocr  \
    --encoder_model_name_or_path microsoft/beit-base-patch16-224 \
    --decoder_model_name_or_path xlm-roberta-base \
    --dataset_name /project/6007993/DataBank/OCR_data/Datasets/al/_Ready/AraOCR_dataset \
    --dataset_config_name ADAB \
    --save_dir /home/ahsang/scratch/arocr/checkpoints/ \
    --output_dir /home/ahsang/scratch/arocr/outputs/ \
    --cache_dir /home/ahsang/scratch/arocr/cache/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \

#wandb agent mahsanghani/arocr/

echo "Training ended at $(date)"