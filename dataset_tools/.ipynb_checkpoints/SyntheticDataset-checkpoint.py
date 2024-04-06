import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_labels = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith(".jpg")])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("L")
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace("image", "mask")
        #mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace(".jpg", ".png") 
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            mask = (mask*255).byte() 
        return image, mask