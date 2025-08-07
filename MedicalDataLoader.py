from operator import index
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import torchio as tio
import warnings
from glob import glob
import albumentations as A

warnings.filterwarnings("ignore")

class MedicalDataset(Dataset):
    """ABIDE dataset."""

    def __init__(self, mode, rootdir='./data/', modality="T1", transform=None, image_size=256, augment=True):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.mode = mode
        self.modality = modality
        self.augment = augment
        self.transform = transform
        self.image_size = image_size
        
        ##This might be modified according to your data
        self.image_paths = glob(os.path.join(rootdir, mode, f'*{modality}.png'))
        self.branmask_paths = [path.replace(modality, 'brainmask') for path in self.image_paths]
        if mode == 'test':
            self.segmentation_paths = [path.replace(modality, 'segmentation') for path in self.image_paths]
        else:
            self.segmentation_paths = []
            
        if len(self.image_paths)==0 :
            raise Exception('No data found')


        if self.augment:
           self.aug = A.Compose([A.Affine (rotate=5, p=0.5),
                A.Affine (translate_px=int(self.image_size//32), p=0.5),
                A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5),])

    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x

    def __len__(self):
        return len(self.image_paths)
        

    def __getitem__(self, index):
        img = np.array(Image.open(self.image_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        brain_mask = np.array(Image.open(self.branmask_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        img = img.astype(np.uint8)
        img[brain_mask==0] = 0
        brain_mask = (brain_mask>0.0).astype(np.int32)
        if self.mode == 'test':
            segmentation = np.array(Image.open(self.segmentation_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
            segmentation = (segmentation>0.0).astype(np.int32)
        else:
            segmentation = np.zeros_like(img)
            
        if self.augment and self.mode == 'train':
            augmented = self.aug(image=img, mask=brain_mask)
            img = augmented['image']
            brain_mask = augmented['mask']
            img[brain_mask==0] = 0
                
        img = img.astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(img)
        else:
            img = self.transform_volume(img)
            img = (img-0.5)/0.5
        return img, brain_mask.astype(np.float32), segmentation
