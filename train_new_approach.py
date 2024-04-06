import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
import os
from evaluate import evaluate_model_metrics
from tqdm import tqdm
from training_components import generate_loss_and_accuracy_file, compute_validation_loss, plot_examples, visualize_results
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_processing_functions import plot_heatmap_from_confusion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_single_batch(model, images, true_masks, criterion, optimizer, gradient_accumulation_steps, device):
    masks_pred = model(images.to(device, dtype=torch.float))
    true_masks = true_masks.to(device, dtype=torch.long)
    loss = criterion(masks_pred, true_masks.squeeze(1))
    loss = loss / gradient_accumulation_steps  # Gradient accumulation
    loss.backward()
    return loss.item()

def train_epoch(model, dataloader, criterion, optimizer, gradient_accumulation_steps, device, progress_bar):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        images, true_masks = batch
        batch_loss = train_single_batch(model, images, true_masks, criterion, optimizer, gradient_accumulation_steps, device)
        epoch_loss += batch_loss

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

    return epoch_loss

def plot_losses(training_loss, validation_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss', marker='o')
    plt.plot(validation_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    #plt.grid(True)
    plt.show()

def settings_file(model, num_epochs, save_model, learning_rate, 
                  save_dir, weights, checkpoint_interval, early_stopping_patience, gradient_accumulation_steps, 
                  min_stopping_epoch , num_classes, start_epoch, batch_size):
    # Create a settings file to store training parameters
    settings_file = open(save_dir + str(model.name) + "/training_settings.txt", "w")
    settings_file.write(f"Model name: {model.name}\n")
    settings_file.write(f"Number of Epochs: {num_epochs}\n")
    settings_file.write(f"Learning Rate: {learning_rate}\n")
    settings_file.write(f"Save Model: {save_model}\n")
    settings_file.write(f"Save Dir: {save_dir}\n")
    settings_file.write(f"Weights: {weights}\n")
    settings_file.write(f"Checkpoint Interval: {checkpoint_interval}\n")
    settings_file.write(f"Early Stopping Patience: {early_stopping_patience}\n")
    settings_file.write(f"Gradient Accumulation Steps: {gradient_accumulation_steps}\n")
    settings_file.write(f"Minimum Stopping Epoch: {min_stopping_epoch}\n")
    settings_file.write(f"Number of Classes: {num_classes}\n")
    settings_file.write(f"Starting Epoch: {start_epoch}\n")
    settings_file.write(f"Batch size: {batch_size}\n")
    settings_file.close()
    


def train_model(model, num_epochs, dataloader, validation_dataloader, device, save_model=True, learning_rate=0.001, 
                save_dir="./results_systematic_approach/", weights=None, checkpoint_interval=10, 
                early_stopping_patience=None, gradient_accumulation_steps=1, min_stopping_epoch = 10, num_classes=3, 
                start_epoch=0, lr_patience = 3, criterion=None, scheduler_func="plateau", optimizer=None):
    
    if criterion==None:
        criterion = torch.nn.CrossEntropyLoss(weights).to(device)
        
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if save_model:
        save_directory = save_dir + str(model.name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if not os.path.exists(save_directory + "/checkpoints"):
            os.makedirs(save_directory + "/checkpoints")

    settings_file(model, num_epochs, save_model, learning_rate, 
                  save_dir, weights, checkpoint_interval, early_stopping_patience, gradient_accumulation_steps, 
                  min_stopping_epoch , num_classes, start_epoch, batch_size=dataloader.batch_size)
    
    log_data = []

    if scheduler_func=="plateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=0.5)
    elif scheduler_func=="cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-5)

    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    tl = float("nan") 
    vl = float("nan")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}, Tr. Loss: {tl:.4f}, Val. Loss: {vl:.4f}, LR: {lr:.2e}")
        epoch_loss = train_epoch(model, dataloader, criterion, optimizer, gradient_accumulation_steps, device, progress_bar)
        
        
        tl =  epoch_loss / len(dataloader)
        vl = compute_validation_loss(model, criterion, validation_dataloader, device)
        log_data.append((epoch+1, tl, vl, lr))

        scheduler.step(vl)

        if save_model and (epoch % checkpoint_interval ==0) and epoch > 0:
            torch.save(model, save_directory + f"/checkpoints/model_{model.name}_{epoch+1}_epochs.pth")

        if early_stopping_patience:
            if vl < best_val_loss:
                best_val_loss = vl
                best_epoch = epoch+1
                patience_counter = 0
                torch.save(model, save_directory + f"/checkpoints/model_best_checkpoint.pth")
            else:
                patience_counter += 1

            if (patience_counter >= early_stopping_patience) and (epoch>=min_stopping_epoch):
                print(f"Early stopping at epoch {epoch} as validation loss did not improve for {early_stopping_patience} epochs.")
                break

    if best_epoch>0:
        checkpoint =  torch.load(save_directory + f"/checkpoints/model_best_checkpoint.pth")
        model.load_state_dict(checkpoint.state_dict())
        
    log_file = generate_loss_and_accuracy_file(save_directory, num_classes=num_classes)
    for epoch, training_loss, validation_loss, learning_rate in log_data:
        log_file.write(f"{epoch},{training_loss:.4f},{validation_loss:.4f},{learning_rate}; \n")
    log_file.close()
    log_data = np.array(log_data)
    
    plot_losses(log_data[:, 1], log_data[:, 2])
    plt.close()

    confusion = evaluate_model_metrics(model, validation_dataloader, num_classes, device)
    plot_heatmap_from_confusion(confusion)
    plt.title("Confusion matrix")
    plt.savefig(save_directory + "/confusion_matrix.jpg")
    plt.show()
    plt.close()
    
    return log_data, confusion
