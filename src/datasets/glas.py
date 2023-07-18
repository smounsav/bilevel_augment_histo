import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import numpy as np
from haven import haven_utils as hu
from PIL import Image 
import torch

from .augmentations import RandAugment

class Glas:
    def __init__(self, split, transform_lvl, datadir_base, patch_size, patch_extractor, n_samples=None, colorjitter=False, val_transform='identity', resize=None, min_resize=None, netA=None, m=5,n=3):
        path = datadir_base or '/home/smounsav/scratch/datasets/Glas_2015'
        self.name = 'glas'
        self.n_classes = 2
        self.image_size = 430
        self.nc = 3
        self.split = split
        self.resize = resize
        self.min_resize = min_resize

        if self.split in ['train', 'validation']:
            self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
            # img = Image.open(self.samples[0][0])
            size = [430, 430] if resize is None else resize
            inverted_size = (size[1], size[0])  # invert terms: PIL returns image size as (width, height)
            self.patch_extractor = patch_extractor(inverted_size, self.patch_size)
            self.image_size = self.patch_size

        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])

        self.mean = normalize.mean
        self.std = normalize.std

        if self.split == 'train':
            if transform_lvl == 0:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size),
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    # transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    # transforms.ToTensor(),
                    # normalize,
                ])
                if netA is not None:
                    transform.transforms.append(netA)

            elif transform_lvl == 1: 
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                    
                    transforms.RandomCrop(self.image_size, padding=4),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 1.5: 
                transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 2:
                transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                    
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])
            
            elif transform_lvl == 2.5:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                    
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 3:
                transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    # transforms.CenterCrop(224),
                    # transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.5, 2)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])
            
            elif transform_lvl == 5:
                transform = transforms.Compose([
                    RandAugment(n,m),
                ])
            else:
                raise ValueError('only lvls 0, 1, 1.5, 2, 2.5 and 3 and 5 are supported')
        
            if colorjitter:
                transform.transforms.append(
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.25,
                        hue=0.04
                    )
                )
            transform.transforms.append(transforms.ToTensor())
            transform.transforms.append(normalize)

        elif self.split in ['validation', 'test']:
            # identity transform
            transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size),
                    # # transforms.Lambda(lambda x: x.convert("RGB")),
                    # transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

        self.transform = transform
        
        if self.split in ['train', 'validation']:
            fname = '/home/smounsav/scratch/datasets/Glas_2015/glas_trainining.json'

            if not os.path.exists(fname):
                dataset = dset.ImageFolder(root=os.path.join(path, 'training'))
                hu.save_json(fname, dataset.imgs)

            self.imgs = np.array(hu.load_json(fname))
            assert(len(self.imgs) == 85)

        elif self.split =='test':
            fname = '/home/smounsav/scratch/datasets/Glas_2015/glas_testing.json'

            if not os.path.exists(fname):
                dataset = dset.ImageFolder(root=os.path.join(path, 'testing'))
                hu.save_json(fname, dataset.imgs)
            self.imgs = np.array(hu.load_json(fname))    
            assert(len(self.imgs) == 80)

        if n_samples is not None:
            assert n_samples % self.n_classes == 0, 'the number of samples %s must be a multiple of the number of classes %s' % (n_samples, self.n_classes)
            with hu.random_seed(10):
                imgs = np.array(self.imgs)
                n = int(n_samples/self.n_classes) # number of samples per class
                # Extract a balanced subset
                ind = np.hstack([np.random.choice(np.where(imgs[:,1] == l)[0], n, replace=False)
                      for l in np.unique(imgs[:,1])])
                # ind = np.random.choice(imgs.shape[0], n_samples, replace=False)
                
                self.imgs = imgs[ind]

    def get_labels(self):
        if self.split in ['test']:
            return np.array([img[1] for img in self.imgs])
        else:
            return np.repeat(np.array([img[1] for img in self.imgs]), len(self.patch_extractor))

    def __getitem__(self, index):
        if self.split in ['test']:
            image_path, labels = self.imgs[index]
            images_original = Image.open(image_path)
            images_original = images_original.convert('RGB')
            images = self.transform(images_original)

            return {"images":images, 
                    'labels':int(labels), 
                    'meta':{'indices':index}}

        else:
            image_path, label = self.imgs[torch.div(index, len(self.patch_extractor), rounding_mode='trunc')]
            patch_index = index % len(self.patch_extractor)
            images_original = Image.open(image_path)
            images = images_original

            if self.resize is not None:
                images = images.resize(self.resize, resample=Image.LANCZOS)
            elif self.min_resize:
                images = F.resize(images, self.min_resize, interpolation=Image.LANCZOS)

            # extract patch
            patch = self.patch_extractor(images, patch_index)
            patch = self.transform(patch)

            return {"images":patch, 
                    'labels':int(label), 
                    'meta':{'indice':index}}

    def __len__(self):
        if self.split in ['test']:
            return len(self.imgs)
        else:
            return len(self.imgs) * len(self.patch_extractor)