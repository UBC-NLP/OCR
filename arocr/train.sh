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

echo "Training started at $(date)"

python train.py \
    --model_name_or_path arocr  
    --encoder_model_name_or_path facebook/deit-base-distilled-patch16-224
    --decoder_model_name_or_path xlm-roberta-base
    --dataset_name ADAB 
    --save_dir arocr/OCR/arocr/checkpoints/
    --do_train
    --do_eval
    --do_predict
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 8
    --seed 42
    --num_train_epochs 1
    --learning_rate 2e-5
    --predict_with_generate
      
echo "Training ended at $(date)"