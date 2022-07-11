#!/bin/bash
encoders=(
    "facebook/deit-base-distilled-patch16-224"
)
decoders=(
    "xlm-roberta-base" "UBC-NLP/ARBERT" "UBC-NLP/MARBERT" "UBC-NLP/MARBERTv2"
)
dataset=(
    'PATS01' 'IDPL-PFOD' 'OnlineKhatt' 'ADAB' 'alexuw' 'MADBase' 'AHCD'
)
echo "encoders: ${encoders[@]}"
echo "decoders: ${decoders[@]}"
echo "dataset: ${dataset[@]}"

for encoder in "${encoders[@]}"; do
    for decoder in "${decoders[@]}"; do
        for dataset in "${dataset[@]}"; do
            epochs=25
            job_name="$dataset-$decoder-$encoder"
            echo "job_name: $job_name"
            sbatch --time=10:00:00 --account=rrg-mageed --mail-user=gbhatia880@gmail.com --mail-type=ALL --nodes=1 --mem=64G --gres=gpu:v100l:4 --job-name=$job_name --output=out_$job_name.out --error=err_$job_name.err ./train_hf.sh $encoder $decoder $dataset $epochs
        done
    done
done

