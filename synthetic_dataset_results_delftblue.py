import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch.losses as losses

from models_2d import UNet_2D_conv_comm_varying_depth
from dataset_tools import generate_dataset, SyntheticDataset, custom_target_transform, custom_transform
from train_new_approach import train_model
from evaluate import evaluate_model_metrics
from data_processing_functions import plot_heatmap_from_confusion

### Settings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
description = "synthetic_data_experiments_2024_04_03"

# Global variables
SUBDOM_WIDTH = 32

# Model variables
NUM_CLASSES = 3
NUM_CHANNELS = 1 

# Generate the datasets
subdomain_sizes = [2,3,4,6,8,16]

for subdom_size in tqdm(subdomain_sizes):
    img_width = subdom_size * SUBDOM_WIDTH
    img_height = SUBDOM_WIDTH

    generate_dataset(img_width=img_width, img_height=img_height, num_images_train=4000,
                     num_images_test=1000, num_images_validation=1000, circle_radius=4,
                     num_circles=1, line_width=3, max_apart=subdom_size, save_descr="new")
    

    dataloaders_training = []
dataloaders_validation = []
dataloaders_test = []

datasets_training = []
datasets_validation = []
datasets_test = []

# Define transforms (used to transform the loaded data)
transform = custom_transform
target_transform = custom_target_transform

# training variables
val_test_batchsize = 128
train_batchsize = 32

# Define dataloaders an validation dataloaders
for i, subdomain_size in enumerate(subdomain_sizes):
    img_dir = f"./data/synthetic_data_new/synthetic_data_{subdomain_size}_subdomains/images/"
    mask_dir = f"./data/synthetic_data_new/synthetic_data_{subdomain_size}_subdomains/masks/"
    img_dir_validation = f"./data/synthetic_data_new/synthetic_data_{subdomain_size}_subdomains_validation/images/"
    mask_dir_validation = f"./data/synthetic_data_new/synthetic_data_{subdomain_size}_subdomains_validation/masks/"
    img_dir_test = f"./data/synthetic_data_new/synthetic_data_{subdomain_size}_subdomains_test/images/"
    mask_dir_test = f"./data/synthetic_data_new/synthetic_data_{subdomain_size}_subdomains_test/masks/"
    

    # Define the datasets and the dataloaders
    datasets_training.append(SyntheticDataset(img_dir, mask_dir, transform=transform, target_transform=target_transform))
    datasets_validation.append(SyntheticDataset(img_dir_validation, mask_dir_validation, transform=transform, target_transform=target_transform))
    datasets_test.append(SyntheticDataset(img_dir_test, mask_dir_test, transform=transform, target_transform=target_transform))

    dataloaders_training.append(DataLoader(datasets_training[i], batch_size=train_batchsize, shuffle=True))
    dataloaders_validation.append(DataLoader(datasets_validation[i], batch_size=val_test_batchsize, shuffle=False))
    dataloaders_test.append(DataLoader(datasets_test[i], batch_size=val_test_batchsize, shuffle=False))


    
# Training hyperparameters
loss = losses.DiceLoss(mode="multiclass")
num_epochs = 30
learning_rate = 0.005
early_stopping_patience = 4
lr_patience = 2
min_stopping_epoch = 10

# Network hyperparameters
subdomain_size = (SUBDOM_WIDTH, SUBDOM_WIDTH)
complexity = 4
dropout_rate = 0.1
kernel_size = 5                     # kernel size of comm network
padding = (kernel_size - 1) // 2    # padding for comm network


# Variables
depths = [3]                                # Depth of U-Net (Number of encoder/decoder blocks)
num_of_feature_maps = [0,1,2,4,8,16,32]     # Number of communicated feature maps

# Initialize results dict
models = {
    "w/ communication": {},
    "baseline": {}
}


for sdomsize in subdomain_sizes:
    models["w/ communication"][sdomsize] = {}
    for d in depths:
        models["w/ communication"][sdomsize][d] = {}
        for nf in num_of_feature_maps:
            models["w/ communication"][sdomsize][d][nf] = {}

            sizex, sizey = (subdomain_size[0] // (2**d), subdomain_size[1] // (2**d))
            inp_size_baseline = (SUBDOM_WIDTH*32, SUBDOM_WIDTH)
            
            unet_comm = UNet_2D_conv_comm_varying_depth(n_channels=NUM_CHANNELS, n_classes=NUM_CLASSES, inp_size = subdomain_size,
                                                        inp_comm = nf, outp_comm=nf, device=device, sizex = sizex, sizey=sizey,
                                                        depth=d, bilinear=False, comm=(False if nf == 0 else True), n_complexity=complexity, dropout_rate=dropout_rate,
                                                        kernel_size=kernel_size, padding=padding, communicator_type=None).to(device)
            
            # unet_baseline = UNet_2D_conv_comm_varying_depth(n_channels=NUM_CHANNELS, n_classes=NUM_CLASSES, inp_size=inp_size_baseline, inp_comm=0, outp_comm=0,
            #                                             device=device, sizex=0, sizey=0, depth=d, bilinear=False, comm=False, n_complexity=complexity,
            #                                             dropout_rate=dropout_rate, kernel_size=0, padding=0).to(device)

            models["w/ communication"][sdomsize][d][nf] = unet_comm
                                                    

def train_model_and_save_results(model, dataloader_train, dataloader_val, save_dir):
    res = train_model(model=model, num_classes=NUM_CLASSES, num_epochs=num_epochs, dataloader=dataloader_train,
                validation_dataloader=dataloader_val,device=device, save_model=True, learning_rate=learning_rate, 
                save_dir=save_dir, weights=None, checkpoint_interval=1000, early_stopping_patience=early_stopping_patience,
                gradient_accumulation_steps=1, min_stopping_epoch=min_stopping_epoch, start_epoch=0, 
                lr_patience=lr_patience, criterion=loss)
    
    return res

for idx, sdomsize in enumerate(subdomain_sizes):
    for d in depths:
        for nf in num_of_feature_maps:
            # define save directory basepath
            save_dir_base = f"./results/synthetic_data/{sdomsize}_subdomains/depth_{d}/with_comm/{nf}_fmaps/"
            
            # define dataloaders
            dataloader_train = dataloaders_training[idx]
            dataloader_val = dataloaders_validation[idx]

            # load model and train
            unet_comm = models["w/ communication"][sdomsize][d][nf]
            res_comm = train_model_and_save_results(model=unet_comm, dataloader_train=dataloader_train, 
                                                    dataloader_val=dataloader_val, save_dir=save_dir_base)