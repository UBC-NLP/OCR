#!/bin/bash
models=(
    "../models/deit-xlmr" 
    "../models/deit-arbert"
    "../models/deit-marbert"
    "../models/deit-marbertv2"
)
dataset=(
    'PATS01' 'IDPL-PFOD' 'shotor' 'UPTI'
)
echo "models: ${models[@]}"
echo "dataset: ${dataset[@]}"

for model_name in "${models[@]}"; do
        for dataset in "${dataset[@]}"; do
            epochs=25
            echo "model_name: $model_name"
            echo "dataset: $dataset"
            echo "epochs: $epochs"
            sbatch ./train_hf.sh $model_name $dataset $epochs
        done
done

