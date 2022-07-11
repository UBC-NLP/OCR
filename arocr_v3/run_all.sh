#!/bin/bash
module load python/3.8 scipy-stack gcc arrow cuda cudnn

source ~/ocr/bin/activate

export NPROC_PER_NODE=1
export NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING='FALSE'
export PARENT=`/bin/hostname -s`
export MPORT=13001
export CHILDREN=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT`
export HOSTLIST="$PARENT $CHILDREN"
export WORLD_SIZE=$SLURM_NTASKS
echo $HOSTLIST

./train_hf.sh $1 $2 $3 $4