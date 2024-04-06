import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tqdm import tqdm

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

            predicted_labels = model(images.to(device, dtype=torch.float))
            predicted_labels = torch.argmax(predicted_labels, dim=1)
            
            true_labels_list = true_labels.cpu().numpy().ravel()
            predicted_labels_list = predicted_labels.cpu().numpy().ravel()

            confusion += confusion_matrix(true_labels_list, predicted_labels_list, labels=range(num_classes))

            torch.cuda.empty_cache()

    return confusion

# def evaluate_model_metrics(model, dataloader, num_classes, device):
#     model.eval()
#     true_labels_list = []
#     predicted_labels_list = []

#     with torch.no_grad():
#         for batch in dataloader:
#             images, true_labels = batch
#             true_labels = true_labels.to(device, dtype=torch.long)
#             true_labels = true_labels[:, 0, :, :]

#             predicted_labels = model(images.to(device, dtype=torch.float))
#             predicted_labels = torch.argmax(predicted_labels, dim=1)
            
#             true_labels_list.extend(true_labels.cpu().numpy().ravel())
#             predicted_labels_list.extend(predicted_labels.cpu().numpy().ravel())

#     #accuracy = accuracy_score(true_labels_list, predicted_labels_list)
#     #precision = precision_score(true_labels_list, predicted_labels_list, average='macro')
#     #recall = recall_score(true_labels_list, predicted_labels_list, average='macro')

#     confusion = confusion_matrix(true_labels_list, predicted_labels_list)
#     #class_recalls = recall_score(true_labels_list, predicted_labels_list, average=None)
#     #class_precisions = precision_score(true_labels_list, predicted_labels_list, average=None)

#     # Remove redundant lists
#     del true_labels_list
#     del predicted_labels_list

#     return confusion

def evaluate_model_metrics_new(model, dataloader, num_classes, device):
    model.eval()
    true_labels_list = []
    predicted_labels_list = []

    if num_classes==2:
        confusion = np.zeros((2,2))
    
    with torch.no_grad():
        for batch in dataloader:
            images, true_labels = batch
            true_labels = true_labels.to(device, dtype=torch.long)
            true_labels = true_labels[:, 0, :, :]

            predicted_labels = model(images.to(device, dtype=torch.float))
            predicted_labels = torch.argmax(predicted_labels, dim=1)

            if num_classes == 2:
                tp = torch.sum((true_labels == 1) & (predicted_labels == 1))
                fp = torch.sum((true_labels == 0) & (predicted_labels == 1))
                fn = torch.sum((true_labels == 1) & (predicted_labels == 0))
                tn = torch.sum((true_labels == 0) & (predicted_labels == 0))

                confusion[0][0] += tn
                confusion[0][1] += fp
                confusion[1][0] += fn
                confusion[1][1] += tp
                
                
            else:
                true_labels_list.extend(true_labels.cpu().numpy().ravel())
                predicted_labels_list.extend(predicted_labels.cpu().numpy().ravel())

    if num_classes == 2:
        confusion
    else:
        confusion = confusion_matrix(true_labels_list, predicted_labels_list)
    #class_recalls = recall_score(true_labels_list, predicted_labels_list, average=None)
    #class_precisions = precision_score(true_labels_list, predicted_labels_list, average=None)

    # Remove redundant lists
    del true_labels_list
    del predicted_labels_list

    return confusion

    

    #accuracy = accuracy_score(true_labels_list, predicted_labels_list)
    #precision = precision_score(true_labels_list, predicted_labels_list, average='macro')
    #recall = recall_score(true_labels_list, predicted_labels_list, average='macro')

    confusion = confusion_matrix(true_labels_list, predicted_labels_list)
    #class_recalls = recall_score(true_labels_list, predicted_labels_list, average=None)
    #class_precisions = precision_score(true_labels_list, predicted_labels_list, average=None)

    # Remove redundant lists
    del true_labels_list
    del predicted_labels_list

    return confusion


# def calculate_recall_precision(true_positive, true_negative, false_positive, false_negative):
#     # Calculate recall (sensitivity)
#     if (true_positive + false_negative) != 0.0:
#         recall = true_positive / (true_positive + false_negative)
#     else:
#         recall = 0.0
    
#     # Calculate precision
#     if (true_positive + false_positive) != 0.0:
#         precision = true_positive / (true_positive + false_positive)
#     else:
#         precision = 0.0
    
#     return recall, precision
    
# def evaluate_model_metrics(model, dataloader, num_classes, device):
#     model.eval()  # Set the model to evaluation mode
#     class_true_positive = [0] * num_classes
#     class_true_negative = [0] * num_classes
#     class_false_positive = [0] * num_classes
#     class_false_negative = [0] * num_classes
    
#     with torch.no_grad():
#         for batch in dataloader:
#             images, true_labels = batch
#             true_labels = true_labels.to(device, dtype=torch.long)
#             true_labels = true_labels[:,0,:,:]
            
#             predicted_labels = model(images.to(device, dtype=torch.float))
#             predicted_labels = torch.argmax(predicted_labels, dim=1)

#             torch.cuda.empty_cache()
            
#             for class_idx in range(num_classes):
#                 true_positive = torch.logical_and(predicted_labels == class_idx, true_labels == class_idx).sum().item()
#                 true_negative = torch.logical_and(predicted_labels != class_idx, true_labels != class_idx).sum().item()
#                 false_positive = torch.logical_and(predicted_labels == class_idx, true_labels != class_idx).sum().item()
#                 false_negative = torch.logical_and(predicted_labels != class_idx, true_labels == class_idx).sum().item()
                
#                 class_true_positive[class_idx] += true_positive
#                 class_true_negative[class_idx] += true_negative
#                 class_false_positive[class_idx] += false_positive
#                 class_false_negative[class_idx] += false_negative
    
#     class_recalls    = []
#     class_precisions = []
#     for class_idx in range(num_classes):
#         recall, precision = calculate_recall_precision(
#             class_true_positive[class_idx], class_true_negative[class_idx],
#             class_false_positive[class_idx], class_false_negative[class_idx]
#         )
#         class_recalls.append(recall)
#         class_precisions.append(precision)

#     accuracy = (sum(class_true_negative) + sum(class_true_positive)) / (sum(class_true_positive) + sum(class_true_negative) + sum(class_false_positive) + sum(class_false_negative))
    
#     return accuracy, class_recalls, class_precisions
