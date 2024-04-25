import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms

class DeepGlobeDatasetMultipleGPUs(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None, data_augmentation=None, size=1224):
        self.img_labels = sorted([file for file in os.listdir(image_dir) if file.endswith("0_0.png") and not file.endswith("_mask.png")])
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation
        self.size = size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx].split('_')[0]  # Extract common part of image names
        img_paths =  [os.path.join(self.img_dir, f"{img_name}_sat_{coord}.png") for coord in ['0_0', f'{self.size}_0', f'0_{self.size}', f'{self.size}_{self.size}']]
        mask_paths =  [os.path.join(self.mask_dir, f"{img_name}_sat_{coord}.png") for coord in ['0_0', f'{self.size}_0', f'0_{self.size}', f'{self.size}_{self.size}']]

        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        masks = [Image.open(mask_path).convert('L') for mask_path in mask_paths]

        if self.data_augmentation:
            images = [self.data_augmentation(image, mask)[0] for image, mask in zip(images, masks)]

        if self.transform:
            images = [self.transform(image) for image in images]

        if self.target_transform:
            masks = [self.target_transform(mask) for mask in masks]
            
        # Concatenate the first two masks column-wise
        first_two_masks_column_wise = torch.cat([masks[0], masks[2]], dim=2)
        # Concatenate the last two masks column-wise
        last_two_masks_column_wise = torch.cat([masks[1], masks[3]], dim=2)
        # Concatenate the two concatenated masks row-wise
        concatenated_mask = torch.cat([first_two_masks_column_wise, last_two_masks_column_wise], dim=1)


        return images, concatenated_mask
    
def custom_target_transform(mask):
    mask = (np.array(mask)).astype(np.uint8) # Convert to float and normalize
    mask = np.expand_dims(mask, axis=0)  # Add channel dimensio
    mask = torch.from_numpy(mask)
    return mask

def custom_transform(image):
    image = np.array(image)   # Convert to float and normalize
    image = image / 255.0
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)
    image = torchvision.transforms.functional.normalize(image, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return image