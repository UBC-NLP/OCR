#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
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
# pip3 install datasets
# pip3 install transformers
# pip3 install torch
# pip3 install wandb

pwd

echo "Training started at $(date)"

python train.py \
    --model_name_or_path microsoft/trocr-base-handwritten  \
    --encoder_model_name_or_path facebook/deit-base-distilled-patch16-224 \
    --decoder_model_name_or_path xlm-roberta-base \
    --dataset_name /home/ahsang/scratch/AraOCR_dataset \
    --dataset_config_name ADAB \
    --save_dir /home/ahsang/scratch/arocr/checkpoints/ \
    --output_dir /home/ahsang/scratch/arocr/outputs/ \
    --cache_dir /home/ahsang/scratch/arocr/cache2/ \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 5 \
    --learning_rate 4.5e-6 \
    --split 0.5

# wandb agent mahsanghani/arocr/

echo "Training ended at $(date)"
