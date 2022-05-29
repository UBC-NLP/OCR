import pandas as pd
from datasets import Dataset, Image, DatasetDict
import argparse


def read_image_file(example):
    with open(example["image"].filename, "rb") as f:
        return {"image": {"bytes": f.read()}}


def get_dataset(root_dir, dataset_name, save_dir, train_split, test_split, specific_name):
    if specific_name:
        path = root_dir + "/" + dataset_name + "/" + specific_name + '.tsv'
    else:
        path = root_dir + "/" + dataset_name + "/" + dataset_name + '.tsv'
    if test_split:
        path1 = root_dir + "/" + dataset_name + "/" + train_split + '.tsv'
        path2 = root_dir + "/" + dataset_name + "/" + test_split + '.tsv'
        df = pd.read_csv(path1, sep='\t')
        df += pd.read_csv(path2, sep='\t')
    else:
        df = pd.read_csv(path, sep='\t')
    df['file_name'] = '' + df['file_name']
    df = df.astype(str)
    dataset = Dataset.from_dict({"image": df['file_name'].to_list()}).cast_column("image", Image())
    dataset = dataset.add_column("text", df['text'].to_list())
    dataset = dataset.map(read_image_file)
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    saved_ds = train_test_valid_dataset.save_to_disk(save_dir)
    return saved_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="ocr", type=str)
    parser.add_argument("--train_split", default="ocr", type=str)
    parser.add_argument("--test_split", default="", type=str)
    parser.add_argument("--specific_name", default="", type=str)
    parser.add_argument("--root_dir", default="data", type=str)
    parser.add_argument("--save_dir", default="data/ocr", type=str)
    args = parser.parse_args()
    print(args.root_dir, args.dataset_name, args.save_dir) 
