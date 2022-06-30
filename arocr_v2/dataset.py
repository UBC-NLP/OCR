import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class OCRDataset(Dataset):
    def __init__(self, data, processor, split, max_target_length, limit_size=None, augment=False, skip_packages=None):
        self.processor = processor
        self.max_target_length = max_target_length
        self.data = data[split]

        print(f'Initializing dataset {split}...')

        print(f'Dataset {split}: {len(self.data)}')

        self.augment = augment
        self.transform_medium, self.transform_heavy = self.get_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'][idx]
        image = self.data['image'][idx]

        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(['none', 'medium', 'heavy'],
                                                 p=[1 - medium_p - heavy_p, medium_p, heavy_p])
            transform = {
                'none': None,
                'medium': self.transform_medium,
                'heavy': self.transform_heavy,
            }[transform_variant]
        else:
            transform = None

        pixel_values = self.read_image(self.processor, image, transform)
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length,
                                          truncation=True).input_ids
        labels = np.array(labels)
        # important: make sure that PAD tokens are ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        encoding = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }
        return encoding

    @staticmethod
    def read_image(processor, path, transform=None):
        img = np.array(path)

        if transform is None:
            transform = A.ToGray(always_apply=True)

        img = transform(image=img)['image']

        pixel_values = processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()

    @staticmethod
    def get_transforms():
        t_medium = A.Compose([
            A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.05),

            A.OneOf([
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise((50, 200), p=0.3),
            A.ImageCompression(0, 30, p=0.1),
            A.ToGray(always_apply=True),
        ])

        t_heavy = A.Compose([
            A.Rotate(10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.05),

            A.OneOf([
                A.Downscale(0.1, 0.2, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.1, 0.2, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur((4, 9), p=0.5),
            A.Sharpen(p=0.5),
            A.RandomBrightnessContrast(0.8, 0.8, p=1),
            A.GaussNoise((1000, 10000), p=0.3),
            A.ImageCompression(0, 10, p=0.5),
            A.ToGray(always_apply=True),
        ])

        return t_medium, t_heavy
        

if __name__ == '__main__':
    from .get_model import get_processor
    from .utils import tensor_to_image

    encoder_name = 'facebook/deit-tiny-patch16-224'
    decoder_name = 'cl-tohoku/bert-base-japanese-char-v2'

    max_length = 300

    dataset = load_dataset("gagan3012/OnlineKhatt")

    processor = get_processor(encoder_name, decoder_name)
    ds = OCRDataset(dataset, processor, 'train', max_length, augment=True)

    for i in range(20):
        sample = ds[i]
        img = tensor_to_image(sample['pixel_values'])
        tokens = sample['labels']
        tokens[tokens == -100] = processor.tokenizer.pad_token_id
        text = ''.join(processor.decode(tokens, skip_special_tokens=True).split())

        print(f'{i}:\n{text}\n')
        plt.imshow(img)
        plt.show()
