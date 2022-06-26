#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:v100l:4
#SBATCH --mail-user=ghaniahsan@outlook.com
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --job-name=run_all
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn
source ~/ocr/bin/activate

echo "Training started at $(date)"

wandb agent mahsanghani/arocr/9p2cqt6e
      
echo "Training ended at $(date)"