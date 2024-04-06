import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import transforms
import torchvision
import torch.nn as nn
import numpy as np

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None, data_augmentation = None):
        self.img_labels = sorted([file for file in os.listdir(image_dir) if file.endswith(".png") and not file.endswith("_mask.png")])
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace("_sat.jpg", "_mask.png").replace(".tiff", ".tif")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.data_augmentation:
            image, mask = self.data_augmentation(image, mask)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

class LungDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_labels = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

        # Define transformations to apply to images and masks
    def custom_transform(self, image):
        #image = image.resize((64, 32))
        image = np.array(image)   # Convert to float and normalize
        image = image / np.max(image)
        #print(image.shape)
        
        image = np.expand_dims(image, axis=0)
        #image = np.transpose(image, (0, 1, 2))  # Change channel order to C x H x
        image = torch.from_numpy(image)
        return image
    
    def custom_target_transform(self, mask):
        #mask = mask.resize((64, 32))
        mask = np.array(mask) / 255.0  # Convert to float and normalize
        #print(mask.shape)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        #mask = np.transpose(mask, (0, 1, 2))  # Change channel order to C x H x
        mask = torch.from_numpy(mask)
        return mask

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("L")
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx])
        #mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace(".jpg", ".png") 
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.custom_transform(image)
        if self.target_transform:
            mask = self.custom_target_transform(mask)
            mask = (mask*1.0).byte() 
        return image, mask

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_transform=None, data_augmentation=False):
        self.img_labels = sorted([os.path.join(root, file) for root, dirs, files in os.walk(image_dir) for file in files if file.endswith('.png')])
        np.random.shuffle(self.img_labels)
        #print(self.img_labels)
        self.img_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.img_labels)

        # Define transformations to apply to images and masks
    def custom_target_transform(self, mask):
        mask = (np.array(mask) / 255.0).astype(np.uint8) # Convert to float and normalize
        #print(mask.shape)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimensio
        mask = torch.from_numpy(mask)
        
        return mask
    
    def custom_transform(self, image):
        image = np.array(image)   # Convert to float and normalize
        image = image / 255.0
        #print(image.shape)
        #image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        #image = transforms.functional.normalize(image, mean=mean, std=std)
        return image
        
    def __getitem__(self, idx):
        img_path = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        mask_path = img_path.replace("/image", "/mask").replace("train_compressed", "train_masks_compressed")
        mask = Image.open(mask_path).convert("L")

        #resize_transform = transforms.Resize((128, 256), interpolation = Image.NEAREST)
        #image = resize_transform(image)
        #mask = resize_transform(mask)
        if self.data_augmentation:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(128, 256))
            image = transforms.functional.crop(image, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
    
            if transforms.RandomHorizontalFlip():
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
    
            if transforms.RandomVerticalFlip():
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        #angle = transforms.RandomRotation.get_params([-10, 10])
        #image = transforms.functional.rotate(image, angle)
        #mask = transforms.functional.rotate(mask, angle)

        if self.transform:
            image = self.custom_transform(image)

        if self.target_transform:
            mask = self.custom_target_transform(mask)

        return image, mask

class CityScapeDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_transform=None):
        self.img_labels = sorted([os.path.join(root, file) for root, dirs, files in os.walk(image_dir) for file in files if file.endswith('.png')])
        self.img_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
        
    def custom_target_transform(self, mask):
        mask = np.array(mask)  # Convert to float and normalize
        #print(mask.shape)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        #mask = np.transpose(mask, (0, 1, 2))  # Change channel order to C x H x
        mask = torch.from_numpy(mask)
        return mask

    #Cityscapes Mean: [0.28692877 0.32516809 0.28392662]
    #Cityscapes STD: [0.16937165 0.17405657 0.17147706]
    
    # Define transformations to apply to images and masks
    def custom_transform(self, image, mean=[0.28692877, 0.32516809, 0.28392662], std= [0.16937165, 0.17405657, 0.17147706]):
        image = np.array(image)   # Convert to float and normalize
        image = image / 255.0
        image = (image - mean) / std
        #print(image.shape)
        #image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        return image

    def __getitem__(self, idx):
        img_path = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        mask_path = img_path.replace("image", "mask")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.custom_transform(image)

        if self.target_transform:
            mask = self.custom_target_transform(mask)

        return image, mask