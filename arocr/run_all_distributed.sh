#!/bin/bash
encoders = (
    "facebook/deit-base-distilled-patch16-224"
)
decoders = (
    "xlm-roberta-base",
    "UBC-NLP/ARBERT",
    "UBC-NLP/MARBERT",
    "UBC-NLP/MARBERTv2",
)
dataset = (
    'PATS01', 'IDPL-PFOD', 'OnlineKhatt', 'ADAB', 'alexuw', 'MADBase', 'AHCD'
)

for encoder in "${encoders[@]}"; do
    for decoder in "${decoders[@]}"; do
        for dataset in "${dataset[@]}"; do
            epochs = 25
            job_name = "$encoder-$decoder-$dataset"
            sbatch --job-name=$job_name --account=def-mageed --mail-user=ghaniahsan@outlook.com --mail-type=ALL --time=5:00:00 --nodes=1 --ntasks-per-node=1 --mem=64G --gres=gpu:v100l:4 --output=out_%x.out --error=err_%x.err ./train_hf.sh $encoder $decoder $dataset $epochs
        done
    done
done
