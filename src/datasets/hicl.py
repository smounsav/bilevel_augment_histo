import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from haven import haven_utils as hu
from PIL import Image 
from multiprocessing import Pool
import warnings
from functools import partial
import csv

from .augmentations import RandAugment

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
        class_name = f[0].split('/')[3]
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

class Hicl:
    def __init__(self, split, transform_lvl, datadir_base, folds_path, fold, patch_size, patch_extractor, organ, magnifying_factor, stain_type, n_samples=None, colorjitter=False, val_transform='identity', resize=None, min_resize=None, m=5,n=3):
        path = datadir_base or '/home/smounsav/scratch/datasets/'
       
        self.resize = resize
        self.min_resize = min_resize

        assert organ in ['brain', 'larynx', 'breast'], 'organ unknown, only brain, breast and larynx are supported'
        if organ == 'brain':
            self.n_classes = 7

            if magnifying_factor == 20:
                fname = path + 'MedispImageLibrary/Brain/20/hicl.json'
                if not os.path.exists(fname):
                    dataset = dset.ImageFolder(root=path + 'MedispImageLibrary/Brain/20')
                    hu.save_json(fname, dataset.imgs)
                self.imgs = np.array(hu.load_json(fname))
                assert(len(self.imgs) == 1257)

                folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Brain/20/folds'
                self.hicl_classes = {'I': 0, 'I-II': 1, 'II': 2, 'II-III': 3, 'III': 4, 'III-IV':5, 'IV':6}

                self.image_size = [1728, 1286]


            elif magnifying_factor == 40:
                fname = path + '/MedispImageLibrary/Brain/40/hicl.json'
                if not os.path.exists(fname):
                    dataset = dset.ImageFolder(root=path + '/MedispImageLibrary/Brain/40')
                    hu.save_json(fname, dataset.imgs)
                self.imgs = np.array(hu.load_json(fname))
                assert(len(self.imgs) == 1297)
                
                folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Brain/40/folds'           
                self.hicl_classes = {'I': 0, 'I-II': 1, 'II': 2, 'II-III': 3, 'III': 4, 'III-IV':5, 'IV':6}
            
                self.image_size = [1728, 1286]

            else:
                raise ValueError('only magnifying factors 20 and 40 are supported for brain cancer images dataset')

        elif organ == 'breast':
            self.n_classes = 3
            if magnifying_factor == 20:
                fname = path + '/MedispImageLibrary/Breast/20/hicl.json'
                if not os.path.exists(fname):
                    dataset = dset.ImageFolder(root=path + '/MedispImageLibrary/Breast/20')
                    hu.save_json(fname, dataset.imgs)
                self.imgs = np.array(hu.load_json(fname))
                assert(len(self.imgs) == 231)
            
                folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Breast/20/folds'
                self.hicl_classes = {'I': 0, 'II': 1, 'III': 2}

            elif magnifying_factor == 40:
                if stain_type == 'HE':
                    fname = path + '/MedispImageLibrary/Breast/40/hicl.json'
                    if not os.path.exists(fname):
                        dataset = dset.ImageFolder(root=path + '/MedispImageLibrary/Breast/40')
                        hu.save_json(fname, dataset.imgs)
                    self.imgs = np.array(hu.load_json(fname))
                    assert(len(self.imgs) == 227)

                    folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Breast/40/folds'
                    self.hicl_classes = {'I': 0, 'II': 1, 'III': 2}

                elif stain_type == 'IHC':
                    fname = path + '/MedispImageLibrary/Breast/ER/40/hicl.json'
                    if not os.path.exists(fname):
                        dataset = dset.ImageFolder(root=path + '/MedispImageLibrary/Breast/ER/40')
                        hu.save_json(fname, dataset.imgs)
                    self.imgs = np.array(hu.load_json(fname))
                    assert(len(self.imgs) == 414)

                    folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Breast/ER/40/folds'
                    self.hicl_classes = {'I': 0, 'II': 1, 'III': 2}

            else:
                raise ValueError('only magnifying factors 20 and 40 are supported for breast cancer images dataset')

        elif organ == 'larynx':
            self.n_classes = 3
            if magnifying_factor == 20:
                fname = path + '/MedispImageLibrary/Larynx/20/hicl.json'
                if not os.path.exists(fname):
                    dataset = dset.ImageFolder(root=path + '/MedispImageLibrary/Larynx/20')
                    hu.save_json(fname, dataset.imgs)
                self.imgs = np.array(hu.load_json(fname))
                assert(len(self.imgs) == 224)

                folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Larynx/20/folds'
                self.hicl_classes = {'I': 0, 'II': 1, 'III': 2}

                self.image_size = [1728, 1296]

            elif magnifying_factor == 40:
                fname = path + '/MedispImageLibrary/Larynx/40/hicl.json'
                if not os.path.exists(fname):
                    dataset = dset.ImageFolder(root=path + '/MedispImageLibrary/Larynx/40')
                    hu.save_json(fname, dataset.imgs)
                self.imgs = np.array(hu.load_json(fname))
                assert(len(self.imgs) == 226)
                
                folds_path = folds_path or '/home/smounsav/scratch/datasets/MedispImageLibrary/Larynx/40/folds'
                self.hicl_classes = {'I': 0, 'II': 1, 'III': 2}

                self.image_size = [1300, 1030]

            else:
                raise ValueError('only magnifying factors 20 and 40 are supported for larynx cancer images dataset')
        
        self.name = 'hicl'
        self.split = split
        self.nc = 3

        if self.split in ['train', 'validation']:
            files = np.array(get_files(folds_path, fold, 'train'))
        else:
            files = np.array(get_files(folds_path, fold, split))
        files_classes = np.array(decode_classes(files, self.hicl_classes))
        self.samples = np.array(check_files(path, files_classes))

        if self.split in ['train', 'validation']:
            self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
            img = Image.open(self.samples[0][0])
            size = img.size if resize is None else resize
            inverted_size = (size[1], size[0])  # invert terms: PIL returns image size as (width, height)
            self.patch_extractor = patch_extractor(inverted_size, self.patch_size)
            self.image_size = self.patch_size

        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])

        self.mean = normalize.mean
        self.std = normalize.std

        if split == 'train':
            if transform_lvl == 0:
                transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    # transforms.ToTensor(),
                    # normalize,
                ])
            
        #     # elif transform_lvl == 1: 
        #     #     transform = transforms.Compose([
        #     #         transforms.RandomCrop(self.image_size, padding=4),
        #     #         transforms.ToTensor(),
        #     #         normalize,
        #     #     ])

        #     # elif transform_lvl == 1.5: 
        #     #     transform = transforms.Compose([
        #     #         transforms.Lambda(lambda x: x.convert("RGB")),
        #     #         transforms.RandomHorizontalFlip(),
        #     #         transforms.ToTensor(),
        #     #         normalize,
        #     #     ])

        #     # elif transform_lvl == 2:
        #     #     transform = transforms.Compose([
        #     #         transforms.Lambda(lambda x: x.convert("RGB")),
        #     #         transforms.RandomCrop(self.image_size, padding=4),
        #     #         transforms.RandomHorizontalFlip(),
        #     #         transforms.ToTensor(),
        #     #         normalize,
        #     #     ])

        #     # elif transform_lvl == 2.5:
        #     #     transform = transforms.Compose([
        #     #         transforms.RandomCrop(self.image_size, padding=4),
        #     #         transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
        #     #         transforms.ToTensor(),
        #     #         normalize,
        #     #     ])

            elif transform_lvl == 3:
                transform = transforms.Compose([
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
                        # brightness=0.5,
                        # contrast=0.5,
                        # saturation=0.5,
                        # hue=0.05                 
                    )
                )
            transform.transforms.append(transforms.ToTensor())
            transform.transforms.append(normalize)

        elif split in ['validation', 'test']:
            # identity transform
            if val_transform == 'identity':
                transform = transforms.Compose([
                        # transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        normalize
                    ])
        #     elif val_transform == 'rotation':
        #         transform = transforms.Compose([
        #                 transforms.Lambda(lambda x: x.convert("RGB")),
        #                 transforms.RandomRotation((45, 45)),
        #                 transforms.ToTensor(),
        #                 normalize
        #             ])
        #     elif val_transform == 'translation':
        #         transform = transforms.Compose([
        #                 transforms.Lambda(lambda x: x.convert("RGB")),                    
        #                 transforms.Pad((4, 4, 0, 0)),
        #                 transforms.CenterCrop(self.image_size),
        #                 transforms.ToTensor(),
        #                 normalize
        #             ])
        #     elif val_transform == 'zoomin':
        #         transform = transforms.Compose([
        #                 transforms.Lambda(lambda x: x.convert("RGB")),
        #                 transforms.Resize(int(self.image_size * 1.5)),
        #                 transforms.CenterCrop(self.image_size),
        #                 transforms.ToTensor(),
        #                 normalize
        #             ])
        #     elif val_transform == 'zoomout':
        #         transform = transforms.Compose([
        #                 transforms.Lambda(lambda x: x.convert("RGB")),
        #                 transforms.Resize(int(self.image_size * 0.75)),
        #                 transforms.Pad(4),
        #                 transforms.ToTensor(),
        #                 normalize
        #             ])

        self.transform = transform
        
        # fname = 'path' + '/hicl.json'

        # if split in ['train', 'validation']:
        #     fname = '/mnt/projects/bilvlda/dataset/tiny-imagenet-200/tinyimagenet_train.json'

        #     if not os.path.exists(fname):
        #         dataset = dset.ImageFolder(root=os.path.join(path, 'train'))
        #         hu.save_json(fname, dataset.imgs)

        #     self.imgs = np.array(hu.load_json(fname))
        #     assert(len(self.imgs) == 100000)

        # elif split =='test':
        #     fname = '/mnt/projects/bilvlda/dataset/tiny-imagenet-200/tinyimagenet_validation.json'

        #     if not os.path.exists(fname):
        #         dataset = dset.ImageFolder(root=os.path.join(path, 'val'))
        #         hu.save_json(fname, dataset.imgs)
        #     self.imgs = np.array(hu.load_json(fname)) 
        #     assert(len(self.imgs) == 10000)

        # if n_samples is not None:
        #     with hu.random_seed(10):
        #         imgs = np.array(self.imgs)
        #         ind = np.random.choice(imgs.shape[0], n_samples, replace=False)
        #         self.imgs = imgs[ind]

        if n_samples is not None:
            # assert n_samples % self.n_classes == 0, 'the number of samples %s must be a multiple of the number of classes %s' % (n_samples, self.n_classes)
            with hu.random_seed(10):
                samples = np.array(self.samples)
                # n = int(n_samples/self.n_classes) # number of samples per class
                n = np.zeros(self.n_classes)
                for idx in range(len(n)):
                    n[idx] = len(np.where(samples[:,1] == str(idx))[0])
                n = np.rint(n * n_samples)
                # Extract a balanced subset
                # ind = np.hstack([np.random.choice(np.where(samples[:,1] == str(l))[0], n[l], replace=False)
                #       for l in range(len(n))])
                ind = np.concatenate([np.random.choice(np.where(samples[:,1] == str(l))[0], n[l].astype(int), replace=False)
                      for l in range(len(n))])
                # ind = np.random.choice(imgs.shape[0], n_samples, replace=False)
                
                self.samples = samples[ind]

    def get_labels(self):
        if self.split in ['test']:
            return np.array([img[1] for img in self.samples])
        else:
            return np.repeat(np.array([img[1] for img in self.samples]), len(self.patch_extractor))

    def __getitem__(self, index):
        if self.split in ['test']:
            image_path, labels = self.samples[index]
            images_original = Image.open(image_path)
            images = self.transform(images_original)

            return {"images":images, 
                    'labels':int(labels), 
                    'meta':{'indices':index}}
        else:
            image_path, label = self.samples[index // len(self.patch_extractor)]
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
            return len(self.samples)
        else:
            return len(self.samples) * len(self.patch_extractor)