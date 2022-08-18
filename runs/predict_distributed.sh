#!/bin/bash
dataset=(
    'PATS01' 'IDPL-PFOD' 'OnlineKhatt' 'ADAB' 'alexuw' 'MADBase' 'AHCD'
)
echo "dataset: ${dataset[@]}"

for dataset in "${dataset[@]}"; do
    echo "dataset: $dataset"
    sbatch ./predict.sh $dataset
done


