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


source ~/ocr/bin/activate
#pip3 install datasets
#pip3 install transformers 
#pip3 install torch
#pip3 install wandb

dataset=$1
cache_dir=$2
le=$3
batch_size=$4
model_name=$5
output_dir=$6
epochs_num=$7
encoder=$8
decoder=$9
seed=${10}


pwd

echo "Training started at $(date)"

python train.py \
    --model_name_or_path $model_name  \
    --encoder_model_name_or_path $encoder \
    --decoder_model_name_or_path $decoder \
    --dataset_name /project/6007993/DataBank/OCR_data/Datasets/al/_Ready/AraOCR_dataset \
    --dataset_config_name $dataset \
    --save_dir ~/scratch/arocr/checkpoints/ \
    --output_dir ~/scratch/arocr/outputs/ \
    --cache_dir ~/scratch/arocr/cache/ \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --seed 42 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --predict_with_generate 
      
echo "Training ended at $(date)"