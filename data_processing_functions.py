import numpy as np
import os
import re
import torch
import matplotlib.pyplot as plt

def plot_heatmap_from_confusion(confusion_matrix, title=None, rescale=True, colorbar_plot=True):
    # Sample 3x3 confusion matrix (replace this with your data)    
    # Normalize the confusion matrix
    
    if rescale:
        normalized_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    else:
        normalized_matrix = confusion_matrix.astype('int')
    
    num_classes = confusion_matrix.shape[0]
    
    # Define class labels
    class_labels = [f'Class {i}' for i in range(num_classes)]
    
    # Create a heatmap for the normalized matrix
    plt.imshow(normalized_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    
    # Add a colorbar
    if colorbar_plot:
        plt.colorbar()
    
    # Customize the plot
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    
    # Label the axes
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text in each cell with color conditionally set to white or black
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            if normalized_matrix[i, j] > 0.5:
                text_color = 'white'
            else:
                text_color = 'black'
            
            if rescale:
                plt.text(j, i, '{:.2f}'.format(normalized_matrix[i, j]), ha='center', va='center', color=text_color)
            else:
                plt.text(j, i, '{:d}'.format(normalized_matrix[i, j]), ha='center', va='center', color=text_color)
    
    # Display the plot
    if title != None:
        plt.title(title)



    

def define_array_from_string(inp_array, num_classes):
    """
    Necessary for the read_data function

    Args:
        - inp_array (str): string as stored in the .txt data files
        - num_classes (int): number of classes

    Returns:
        - inp_array (np.array)
    """
    inp_array = " ".join(inp_array.split())
    inp_array = inp_array.replace("[", "").replace("]", "")
    inp_array = np.fromstring(inp_array, dtype=int, sep=" ").reshape(num_classes, num_classes)
    return inp_array

def read_data(sourcedir, num_classes = 3):
    """
    Reads a datafile and returns the training metrics stored in this file 

    Args:
        - sourcedir (PATH): path to a .txt datafile
        - num_classes (int): number of classes for the problem

    Returns:
        - epochs (list)          : list with epoch numbers
        - training_loss (list)   : list with training loss for each epoch
        - validation_loss (list) : list with validation loss or each epoch
        - cf_training (list)     : list of arrays with training confusion matrix for each epoch
        - cf_validation (list)   : list of arrays with validation confusion matrix for each epoch
    """

    with open(sourcedir, "r") as file:
        data = file.read()

    data_lines = data.split(";")
    epochs = []
    training_loss = []
    validation_loss = []
    confusion_training = []
    confusion_validation = []

    for line in data_lines:
        # Skip empty lines
        if not line.strip():
            continue

        epoch, tl, vl, tc, vc = line.strip().split(",")
        epochs.append(int(epoch))
        training_loss.append(float(tl))
        validation_loss.append(float(vl))

        tc = define_array_from_string(tc, num_classes)
        vc = define_array_from_string(vc, num_classes)
        
        # Convert the confusion matrices to NumPy arrays
        confusion_training.append(tc)
        confusion_validation.append(vc)

    return epochs, training_loss, validation_loss, confusion_training, confusion_validation

def calculate_metrics(confusion_matrices):
    """
    This function calculates the precision and recall values. It needs as input a list of confusion matrices and returns a list
    of equal length of dimension NUM_EPOCHS x 2 (precision + recall) x NUM_CLASSES
    """
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

def plot_images_and_pred(model, dataloader, device, num_images=5):
    """ 
    Plots images and predictions for RGB input images and L masks
    """
    for batch in dataloader:
        images, masks = batch
        predictions = torch.argmax(model(images.to(device, dtype=torch.float)), dim=1)
        for i in range(min(num_images, len(dataloader))):
            plt.subplot(1,3,1)
            plt.title("Image")
            plt.axis("off")
            plt.imshow(images[i].permute(1,2,0))
            plt.subplot(1,3,2)
            plt.title("Mask")
            plt.axis("off")
            plt.imshow(masks[i,0], vmin=0, vmax=20)
            plt.subplot(1,3,3)
            plt.title("Predicted Mask")
            plt.axis("off")
            plt.imshow(predictions[i].cpu(), vmin=0, vmax=20)

            plt.tight_layout()
            plt.show()
        break

def compute_iou_from_confusion(confusion_matrix, print_iou=True):
    """
    Computes the IoU from the confusion matrix. If print_iou is True, the result is printed afterwards.
    """
    iou_per_class = []
    
    # Compute IoU for each class
    for class_idx in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[class_idx, class_idx]
        fp = np.sum(confusion_matrix[:, class_idx]) - tp
        fn = np.sum(confusion_matrix[class_idx, :]) - tp
    
        iou = tp / (tp + fp + fn)
        iou_per_class.append(iou)
    
    # Print the IoU values for each class
    if print_iou:
        for class_idx, iou in enumerate(iou_per_class):
            print(f"Class {class_idx}: IoU = {iou:.4f}")

    return iou_per_class
