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

#./train_hf.sh ../models/deit-xlmr ADAB 1

model_name=$1
dataset=$2
epochs=$3
SLURM_JOB_NUM_NODES=1
NPROC_PER_NODE=4

module load python/3.8 scipy-stack gcc arrow cuda cudnn

source ~/ENV38_default/bin/activate


echo "Training started at $(date)"
echo "Num of node $SLURM_JOB_NUM_NODES"
echo "Num of GPU per node $NPROC_PER_NODE"
echo "PROCID: $SLURM_PROCID"
echo "LOCALID: $SLURM_LOCALID"
echo "model_name: $model_name"
echo "Dataset: $dataset"
echo "Epochs: $epochs"

echo "Training started at $(date)"

python train.py \
   --model_name_or_path $model_name \
   --dataset_name /home/ahsang/scratch/AraOCR_dataset \
   --dataset_config_name $dataset \
   --output_dir ~/scratch/arocr/results \
   --cache_dir ~/scratch/arocr/cache/ \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --num_train_epochs $epochs \

echo "Training ended at $(date)"
