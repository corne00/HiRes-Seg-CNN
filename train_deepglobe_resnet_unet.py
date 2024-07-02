# Standard library imports
import os
import time
import json
import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from loss_functions import DiceLoss  #, FocalLoss
from models_multiple_GPUs import *
from dataset_tools import DeepGlobeDatasetMultipleGPUs, custom_transform, custom_target_transform, data_augmentation
from data_processing_functions import plot_heatmap_from_confusion, compute_iou_from_confusion

# Parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a multi-GPU UNet model with communication layers.")
    
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training (default: 12).')
    parser.add_argument('--batch_size_test', type=int, default=12, help='Batch size for testing and validation (default: 12).')
    parser.add_argument('--num_comm_fmaps', type=int, default=64, help='Number of communicated feature maps (default: 64).')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model (default: 50).')
    parser.add_argument('--comm', action='store_true', help='Enable communication in the model.')
    parser.add_argument('--save_path', type=str, help='Path to save the trained model. If not provided, a default path will be generated.')
    parser.add_argument('--subdomain_dist', type=int, nargs=2, default=(2, 2), help='Subdomain distribution for model (default: (2, 2)).')
    parser.add_argument('--image_size', type=int, default=1024, help='Size of the input images (default: 2448).')
    parser.add_argument('--image_size_test', type=int, default=1024, help='Size of the input images for testing (default: 2448).')
    parser.add_argument('--exchange_fmaps', action='store_true', help='Exchange feature maps between GPUs (default: True).')
    
    args = parser.parse_args()

    # Build in some checks
    if args.num_comm_fmaps == 0:
        args.comm = False
        args.exchange_fmaps = False

    if not args.comm:
        args.num_comm_fmaps = 0
        args.exchange_fmaps = False

    # Generate a default save_path if not provided
    comm_status = "with_comm" if args.comm else "without_comm"
    fmaps_status = "exchange_fmaps" if args.exchange_fmaps else "no_exchange_fmaps"
    subdomain_folder = f"subdomain_{args.subdomain_dist[0]}x{args.subdomain_dist[1]}"
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    if not args.save_path:
        args.save_path = (
            f"./results/results_{timestamp}/{subdomain_folder}/{comm_status}/{fmaps_status}/fmaps_{args.num_comm_fmaps}_{timestamp}"
        )
    else:
        args.save_path = (
            f"./results/{args.save_path}/{subdomain_folder}/{comm_status}/{fmaps_status}/fmaps_{args.num_comm_fmaps}_{timestamp}"
        )

    return args

# Parse the arguments
args = parse_arguments()

print(args)

# Use the parsed arguments
batch_size = args.batch_size
batch_size_test = args.batch_size_test
num_comm_fmaps = args.num_comm_fmaps
num_epochs = args.num_epochs
comm = args.comm
save_path = args.save_path
subdomain_dist = args.subdomain_dist
image_size = args.image_size
image_size_test = args.image_size_test
exchange_fmaps = args.exchange_fmaps

data_type = torch.float16 if torch.cuda.is_available() else torch.bfloat16

# Create dictories
os.makedirs(save_path, exist_ok=True)

# Save the arguments to a json
def save_args_to_json(args, dir_path):
    # Create a dictionary from the parsed arguments
    args_dict = vars(args)
    
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Define the path for the JSON file
    json_file_path = os.path.join(dir_path, f'training_args.json')

    # Save the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print(f"Arguments saved to {json_file_path}")

save_args_to_json(args, save_path)

# Set devices
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
print("Available GPUs:", devices)

# Check if we have half precision
half_precision = torch.cuda.is_available()

# Half precision scaler
scaler = GradScaler(enabled=half_precision)

### Main code
PS_train = image_size
PS_val = image_size_test
PS_test = image_size_test

# SET DATALOADERS
image_dir_train = f"/scratch/cverburg/kaggle-deepglobe/cropped-{PS_train}/train/images/"
mask_dir_train = f"/scratch/cverburg/kaggle-deepglobe/cropped-{PS_train}/train/gt/"

image_dir_val = f"/scratch/cverburg/kaggle-deepglobe/cropped-{PS_val}/val/images/"
mask_dir_val = f"/scratch/cverburg/kaggle-deepglobe/cropped-{PS_val}/val/gt/"

image_dir_test = f"/scratch/cverburg/kaggle-deepglobe/cropped-{PS_test}/test/images/"
mask_dir_test = f"/scratch/cverburg/kaggle-deepglobe/cropped-{PS_test}/test/gt/"

# Load datasets
imagedataset_train = DeepGlobeDatasetMultipleGPUs(
    image_dir_train, mask_dir_train, 
    transform=custom_transform, target_transform=custom_target_transform, 
    data_augmentation=data_augmentation, subdomains_dist=subdomain_dist
)
imagedataset_val = DeepGlobeDatasetMultipleGPUs(
    image_dir_val, mask_dir_val, 
    transform=custom_transform, target_transform=custom_target_transform, 
    subdomains_dist=subdomain_dist
)
imagedataset_test = DeepGlobeDatasetMultipleGPUs(
    image_dir_test, mask_dir_test, 
    transform=custom_transform, target_transform=custom_target_transform, 
    subdomains_dist=subdomain_dist
)

# Define dataloaders
dataloader_train = DataLoader(imagedataset_train, batch_size=batch_size, shuffle=True, num_workers=6)
dataloader_val = DataLoader(imagedataset_val, batch_size=batch_size_test, shuffle=False, num_workers=6)
dataloader_test = DataLoader(imagedataset_test, batch_size=batch_size_test, shuffle=False, num_workers=6)

# Function to compute validation loss
def compute_validation_loss(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with torch.autocast(device_type = 'cuda' if torch.cuda.is_available() else "cpu", dtype=data_type, enabled=half_precision):
            for images, masks in tqdm(dataloader):
                images = [im.half() for im in images]
                masks = masks.to(device, dtype=torch.long)
    
                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_loss += loss.item()
                num_batches += 1
    
    average_loss = total_loss / num_batches
    return average_loss

# Function to train parallel model
def test_parallel_model(comm=True, num_epochs=25, num_comm_fmaps=64, save_path= f"{save_path}/"):
    
    if num_comm_fmaps == 0:
        comm = False
    
    unet = MultiGPU_ResNetUNet_with_comm(n_channels=3, n_classes=7, input_shape=(PS_train, PS_train), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=4, subdom_dist=subdomain_dist,
                 bilinear=False, comm=comm, complexity=64, dropout_rate=0.1, kernel_size=5, padding=2, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps))
    unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
              
    if comm:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters()) + list(unet.communication_network.parameters()) 
    else:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters())
        
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss = DiceLoss(mode="multiclass", ignore_index=6)
    losses = []

    # Wrap your training loop with tqdm
    start_time = time.time()
    validation_losses = []
    best_val_loss = float('inf')

    # Iterate over the epochs
    for epoch in range(num_epochs):
        unet.train()
        epoch_losses = []  # Initialize losses for the epoch
        
        for images, masks in tqdm(dataloader_train):
            optimizer.zero_grad()
            
            with torch.autocast(device_type = 'cuda' if torch.cuda.is_available() else "cpu", dtype=data_type, enabled=half_precision):
                images = ([im.half() for im in images] if torch.cuda.is_available() else [im.float() for im in images])
                masks = masks.to(devices[0], dtype=torch.long)
    
                ## Forward propagation:
                # Run batch through encoder
                predictions = unet(images)
                
                ## Backward propagation
                l = loss(predictions, masks)
                
            scaler.scale(l).backward()
            
            losses.append(l.item())  # Append loss to global losses list
            epoch_losses.append(l.item())  # Append loss to epoch losses list

            with torch.no_grad():
                for i in range(1, len(unet.encoders)):
                    for param1, param2 in zip(unet.encoders[0].parameters(), unet.encoders[i].parameters()):
                        if param1.grad is not None:
                            param1.grad += param2.grad.to(devices[0])
                            param2.grad = None

            with torch.no_grad():
                for i in range(1, len(unet.decoders)):
                    for param1, param2 in zip(unet.decoders[0].parameters(), unet.decoders[i].parameters()):
                        param1.grad += param2.grad.to(devices[0])
                        param2.grad = None

            
            scaler.step(optimizer)
            scaler.update()

            for i in range(1, len(unet.encoders)):
                unet.encoders[i].load_state_dict(unet.encoders[0].state_dict())
                
            for i in range(1, len(unet.decoders)):
                unet.decoders[i].load_state_dict(unet.decoders[0].state_dict())
            
        
    
        # Compute and print validation loss
        val_loss = compute_validation_loss(unet, loss, dataloader_val, devices[0])
        print(f'Validation Loss (Dice): {val_loss:.4f}')
        
        validation_losses.append(val_loss)
        
        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Track maximum GPU memory used
        max_memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        print(f"Maximum GPU Memory Used in Epoch {epoch+1}: {max_memory_used:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    
    print(f"Training the model {'with' if comm else 'without'} communication network took: {time.time() - start_time:.2f} seconds.")
    
    # Load the best weights
    unet.load_weights(load_path=os.path.join(save_path, "unet.pth"), device=devices[0])
    
    return unet, validation_losses

unet, losses = test_parallel_model(comm=comm, num_comm_fmaps=num_comm_fmaps, num_epochs=num_epochs)

# Plot and save the losses
plt.plot(losses)
plt.savefig(f"{save_path}/loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# After the training loop, save the losses to a JSON file
with open(f"{save_path}/losses.json", "w") as f:
    json.dump(losses, f)

# Define a function to evaluate the model metrics and compute the confusion matrix
def evaluate_model_metrics(model, dataloader, num_classes, device):
    model.eval()
    true_labels_list = []
    predicted_labels_list = []

    confusion = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for batch in tqdm(dataloader, total = len(dataloader)):
            
            
            images, true_labels = batch
            true_labels = true_labels.to(device, dtype=torch.long)
            true_labels = true_labels[:, 0, :, :]

            with torch.autocast(device_type = 'cuda' if torch.cuda.is_available() else "cpu", dtype=data_type, enabled=half_precision):
                predicted_labels = model([image.half() for image in images])
                predicted_labels = torch.argmax(predicted_labels, dim=1)
            
            true_labels_list = true_labels.cpu().numpy().ravel()
            predicted_labels_list = predicted_labels.cpu().numpy().ravel()

            confusion += confusion_matrix(true_labels_list, predicted_labels_list, labels=range(num_classes))

            torch.cuda.empty_cache()

    return confusion

cf = evaluate_model_metrics(model=unet, dataloader=dataloader_test, device=None, num_classes=6)

# Plot the confusion matrix
plot_heatmap_from_confusion(cf)
plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Compute the confusion score
iou_scores = compute_iou_from_confusion(cf) 
average_iou = np.mean(iou_scores)

print(f"Confusion Matrix:\n{cf}")
print(f"Average IoU Score: {average_iou}")

# Save IOU scores, average IOU score, and confusion matrix to a JSON file
results = {
    "image size": image_size,
    "iou_scores": iou_scores,
    "average_iou": average_iou,
    "confusion_matrix": cf.tolist()
}

json_file_path = os.path.join(save_path, f"evaluation.json")
with open(json_file_path, "w") as json_file:
    json.dump(results, json_file)

print(f"Evaluation results saved to: {json_file_path}")
