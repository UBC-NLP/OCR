#!/bin/bash
#SBATCH --time=0-2:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=64G
#SBATCH --account=def-mageed

source ~/.bashrc
source activate ENV39_hug

################

module load gcc arrow
module load python/3.8
source ~/ENV38_default/bin/activate
pip3 install pandas
pip3 install datasets
pip3 install argparse
python3 /scratch/ahsang/dataset.py \
      --dataset_name MADBase \
      --train_split MADBase_Train \
      --test_split MADBase_Test \
      --root_dir /project/6007993/DataBank/OCR_data/Datasets/al/_Ready \
      --save_dir processed/

