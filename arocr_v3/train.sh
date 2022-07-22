#!/bin/bash
/bin/hostname -s
export NCCL_BLOCKING_WAIT=1

encoder=$1
decoder=$2
dataset=$3
epochs=$4

echo "Training started at $(date)"
echo "Num of node $SLURM_JOB_NUM_NODES"
echo "Num of GPU per node $NPROC_PER_NODE"
echo "PROCID: $SLURM_PROCID"
echo "LOCALID: $SLURM_LOCALID"
echo "Encoder: $encoder"
echo "Decoder: $decoder"
echo "Dataset: $dataset"
echo "Epochs: $epochs"

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    train.py \
    --encoder_model_name_or_path $encoder \
    --decoder_model_name_or_path $decoder \
    --dataset_name /home/ahsang/scratch/AraOCR_dataset \
    --dataset_config_name $dataset \
    --save_dir ~/scratch/arocr/checkpoints/ \
    --output_dir ~/scratch/arocr/outputs/ \
    --cache_dir ~/scratch/arocr/cache/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs $epochs \

echo "Training ended at $(date)"