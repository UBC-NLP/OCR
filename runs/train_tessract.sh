#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --account=def-mageed
#SBATCH --job-name=train_tess
#SBATCH --output=train_tess.out
#SBATCH --error=train_tess.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load gcc/9.3.0 arrow python/3.9 scipy-stack tesseract cuda/11.4 opencv

source ~/ENV38_default/bin/activate

python tesseract.py 