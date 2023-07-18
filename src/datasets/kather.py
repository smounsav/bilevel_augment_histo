import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from haven import haven_utils as hu
from PIL import Image 
from multiprocessing import Pool
import warnings
from functools import partial
import csv

def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out


def get_files(folds_path: str, fold: int, split: str) -> list:
    csv_dir = os.path.join(folds_path, 'fold_{}'.format(fold))
    csv_file = os.path.join(csv_dir, '{}_f_{}.csv'.format(split, fold))
    if os.path.exists(csv_file):
        files = csv_reader(csv_file)
    else:
        raise FileExistsError('File {} not found.'.format(csv_file))
    return files


def decode_classes(files: list, classes: dict) -> list:
    files_decoded_classes = []
    for f in files:
        class_name = f[0].split('/')[1]
        files_decoded_classes.append((f[0], classes[class_name]))

    return files_decoded_classes


def load_image(image_path: str, resize=None, min_resize=None) -> tuple:
    image = Image.open(image_path)
    if resize is not None:
        image = image.resize(resize, resample=Image.LANCZOS)
    elif min_resize:
        image = F.resize(image, min_resize, interpolation=Image.LANCZOS)
    return image_path, image


def load_data(samples: list, resize=None, min_resize=None, num_workers: int = None) -> dict:
    load_image_partial = partial(load_image, resize=resize, min_resize=min_resize)
    if num_workers is not None:
        with Pool(num_workers) as p:
            images = p.map(load_image, samples)
    else:
        images = map(load_image_partial, samples)
    images = dict(images)
    return images


def check_file(file: tuple, path: str):
    file_path, label = file
    file_full_path = os.path.join(path, file_path)

    if os.path.isfile(file_full_path):
        return file_full_path, label
    else:
        return None


def check_files(path: str, files: list) -> list:
    if not os.path.isdir(path):
        raise NotADirectoryError('{} is not present.'.format(path))

    check_file_partial = partial(check_file, path=path)
    found_files = map(check_file_partial, files)
    found_files = list(filter(lambda x: x is not None, found_files))

    if len(found_files) != len(files):
        warnings.warn('Only {} image files found out of the {} provided.'.format(len(found_files), len(files)))

    return found_files


class Kather(Dataset):
    kather_classes = {'01_TUMOR': 0, '02_STROMA': 1, '03_COMPLEX': 2, '04_LYMPHO': 3, '05_DEBRIS': 4, '06_MUCOSA':5, '07_ADIPOSE':6, '08_EMPTY': 7}
    def __init__(self, split, transform_lvl, datadir_base, folds_path, fold, n_samples=None, colorjitter=False, val_transform='identity', resize=None, min_resize=None):
        path = datadir_base or '/home/smounsav/scratch/datasets'
        folds_path = folds_path or '/home/smounsav/scratch/datasets/Kather_texture_2016_image_tiles_5000/folds'
        self.name = 'kather'
        self.n_classes = 8
        self.split = split

        self.nc = 3
        self.resize = resize
        self.min_resize = min_resize


        if self.split in ['train', 'validation']:
            files = get_files(folds_path, fold, 'train')
        else:
            files = get_files(folds_path, fold, split)
        files_classes = decode_classes(files, self.kather_classes)
        self.samples = check_files(path, files_classes)
        self.image_size = 150
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])

        self.mean = normalize.mean
        self.std = normalize.std

        if split == 'train':
            if transform_lvl == 0:
                transform = transforms.Compose([])
            
            elif transform_lvl == 1: 
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                ])

            elif transform_lvl == 1.5: 
                transform = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                ])

            elif transform_lvl == 2:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomVerticalFlip(),
                ])

            elif transform_lvl == 2.5:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                ])

            elif transform_lvl == 3:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    transforms.RandomVerticalFlip(),
                ])

            else:
                raise ValueError('only lvls 0, 1, 1.5, 2, 2.5 and 3 are supported')

            if colorjitter:
                transform.transforms.append(
                    transforms.ColorJitter(
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.05
                    ))

            transform.transforms.append(transforms.ToTensor())
            transform.transforms.append(normalize)

        elif split in ['validation', 'test']:
            # identity transform
            if val_transform == 'identity':
                transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'rotation':
                transform = transforms.Compose([
                        transforms.RandomRotation((45, 45)),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'translation':
                transform = transforms.Compose([
                        transforms.Pad((4, 4, 0, 0)),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'zoomin':
                transform = transforms.Compose([
                        transforms.Resize(int(self.image_size * 1.5)),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'zoomout':
                transform = transforms.Compose([
                        transforms.Resize(int(self.image_size * 0.75)),
                        transforms.Pad(4),
                        transforms.ToTensor(),
                        normalize
                    ])

        self.transform = transform
        
    def get_labels(self):
        return np.array([img[1] for img in self.samples])

    def __getitem__(self, index):
        image_path, labels = self.samples[index]
        images_original = Image.open(image_path)
        images = self.transform(images_original)

        return {"images":images, 
                'labels':int(labels), 
                'meta':{'indices':index}}
 
    def __len__(self):
        return len(self.samples)