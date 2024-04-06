import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_loss_and_accuracy_file(save_directory, num_classes):
    # Create a file to write training_loss and accuracy
    log_file_path = os.path.join(save_directory, "loss_and_accuracy.txt")
    log_file = open(log_file_path, "w")
    return log_file

def compute_validation_loss(model, criterion, validation_dataloader, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in validation_dataloader:
            images, true_masks = batch
            
            true_masks = true_masks.to(device, dtype=torch.long)
            masks_pred = model(images.to(device, dtype=torch.float))
            val_loss += criterion(masks_pred, true_masks.squeeze(1)).item()
            torch.cuda.empty_cache()
            
    return val_loss / len(validation_dataloader)

def plot_examples(masks_pred, true_masks, images, save_directory, epoch, batch_idx):
    # Convert masks_pred and true_masks to numpy arrays
    masks_pred_np = np.argmax(masks_pred.detach().squeeze(1).cpu().numpy(), axis=1)
    true_masks_np = true_masks.squeeze(1).cpu().numpy()

    #print(masks_pred_np.shape)
    #print(true_masks_np.shape)
    
    # Plot the predicted and true masks
    plt.subplot(1, 3, 1)
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    #plt.imshow(masks_pred_np[0], alpha=0.5, cmap='jet') 
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(masks_pred_np[0], cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(true_masks_np[0], cmap='gray')
    plt.title("True Mask")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_directory + f"/epoch_{epoch}_batch_{batch_idx}.png", bbox_inches='tight')
    #print(save_directory + f"/epoch_{epoch}_batch_{batch_idx}.png")
    #plt.show()
    plt.close()

def visualize_results(confusion_matrix_training, confusion_matrix_validation, savedir, num_classes=5):
    def calculate_metrics(confusion_matrices):
        metrics_per_epoch = []

        for confusion_matrix in confusion_matrices:
            num_classes = confusion_matrix.shape[0]
            recall = np.zeros(num_classes)
            precision = np.zeros(num_classes)

            for i in range(num_classes):
                tp = confusion_matrix[i, i]
                fn = np.sum(confusion_matrix[i, :]) - tp
                fp = np.sum(confusion_matrix[:, i]) - tp

                recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0

            metrics_per_epoch.append((precision, recall))
        return np.array(metrics_per_epoch)
        
    training_metrics = calculate_metrics(confusion_matrix_training)
    validation_metrics = calculate_metrics(confusion_matrix_validation)
    #print(training_metrics.shape[0])
    epochs = range(0, len(confusion_matrix_training))

    plt.figure(figsize=(9,13))
    plt.subplot(2,1,1)
    for class_num in range(num_classes):
        plt.plot(epochs, training_metrics[:, 0, class_num], marker='o', label=f'Precision (Train, Class {class_num})')
        plt.plot(epochs, validation_metrics[:, 0, class_num], marker='x', label=f'Precision (Validation, Class {class_num})')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
    plt.title(f'Precision')
    plt.xticks(epochs)
    plt.legend(bbox_to_anchor=(1, 1))  # Place legend next to the plot

    plt.subplot(2,1,2)
    for class_num in range(num_classes):
        plt.plot(epochs, training_metrics[:, 1, class_num], marker='o', label=f'Recall (Train, Class {class_num})')
        plt.plot(epochs, validation_metrics[:, 1, class_num], marker='x', label=f'Recall (Validation, Class {class_num})')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
    plt.title(f'Recall')
    plt.xticks(epochs)
    plt.legend(bbox_to_anchor=(1, 1))  # Place legend next to the plot

    plt.savefig(savedir, bbox_inches='tight')
    plt.show()
    plt.close()
    