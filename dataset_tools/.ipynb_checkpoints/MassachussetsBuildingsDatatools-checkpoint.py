import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import random
import os 

def custom_target_transform(mask):
    mask = (np.array(mask) / 255).astype(np.uint8) # Convert to float and normalize
    mask = np.expand_dims(mask, axis=0)  # Add channel dimensio
    mask = torch.from_numpy(mask)
    return mask

def custom_transform(image):
    image = np.array(image)   # Convert to float and normalize
    image = image / 255.0
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)
    return image

def data_augmentation(image, mask):
    # Convert PIL images to numpy arrays for easier manipulation
    image_array = np.array(image)
    mask_array = np.array(mask)

    # Randomly apply horizontal flip
    if random.choice([True, False]):
        image_array = np.fliplr(image_array)
        mask_array = np.fliplr(mask_array)

    # Randomly apply vertical flip
    if random.choice([True, False]):
        image_array = np.flipud(image_array)
        mask_array = np.flipud(mask_array)

    angle = random.uniform(-15, 15)  # Ensure the same angle for both transformations
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array)

    # image = image.rotate(angle, resample=Image.BICUBIC)
    # mask = mask.rotate(angle, resample=Image.NEAREST)
    
    return image, mask

class RoadsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None, augmentations = True, interpolation=None, size=128, random_patch=False):
        self.img_labels = sorted([file for file in os.listdir(image_dir) if file.endswith(".png") or file.endswith(".tiff") and not file.endswith("_mask.png")])
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations
        self.interpolation = interpolation
        self.size = size
        self.random_patch = random_patch

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace("_sat.jpg", "_mask.png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        SIZE = self.size

        if image.size[0] > SIZE or image.size[1] > SIZE:
            i = 0
            j = 0
            if self.random_patch:
                i = random.randint(0, image.size[0] - SIZE)
                j = random.randint(0, image.size[1] - SIZE)
            image = image.crop((i, j, i + SIZE, j + SIZE))
            mask = mask.crop((i, j, i + SIZE, j + SIZE))

        if self.augmentations:
            image, mask = data_augmentation(image, mask)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
     
