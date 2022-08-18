from datasets import load_dataset, concatenate_datasets, interleave_datasets, DatasetDict
import os 
os.environ["HF_DATASETS_OFFLINE"] = 1

adab = load_dataset(
    "/home/gagan30/scratch/arocr/AraOCR_dataset","ADAB", cache_dir="/home/gagan30/scratch/arocr/cache")

khatt = load_dataset(
    "/home/gagan30/scratch/arocr/AraOCR_dataset", "OnlineKhatt", cache_dir="/home/gagan30/scratch/arocr/cache")

alexuw = load_dataset(
    "/home/gagan30/scratch/arocr/AraOCR_dataset", "alexuw", cache_dir="/home/gagan30/scratch/arocr/cache")

merged_train = concatenate_datasets([adab['train'], khatt['train'], alexuw['train']])
merged_valid = interleave_datasets(
    [adab['validation'], khatt['validation'], alexuw['validation']],seed=42)
merged_test = interleave_datasets(
    [adab['test'], khatt['test'], alexuw['test']],seed=42)

dataset = DatasetDict(
    {
        'train': merged_train,
        'valid': merged_valid,
        'test': merged_test
    }
)

print(dataset)


