
import os
import time
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
from models_multiple_GPUs import *
from dataset_tools import DeepGlobeDatasetMultipleGPUs, custom_transform, custom_target_transform
from sklearn.metrics import confusion_matrix

os.makedirs("./figures", exist_ok=True)

PS_train = 1224
PS_val = 1224
PS_test = 1224

image_dir_train = f"/kaggle/input/deepglobe-dataset-new-ps-{PS_train}/cropped-{PS_train}/train/images/"
mask_dir_train = f"/kaggle/input/deepglobe-dataset-new-ps-{PS_train}/cropped-{PS_train}/train/gt/"

image_dir_val   = f"/kaggle/input/deepglobe-dataset-new-ps-{PS_val}/cropped-{PS_val}/val/images/"
mask_dir_val  = f"/kaggle/input/deepglobe-dataset-new-ps-{PS_val}/cropped-{PS_val}/val/gt/"

image_dir_test   = f"/kaggle/input/deepglobe-dataset-new-ps-{PS_test}/cropped-{PS_test}/test/images/"
mask_dir_test  = f"/kaggle/input/deepglobe-dataset-new-ps-{PS_test}/cropped-{PS_test}/test/gt/"

imagedataset_train = DeepGlobeDatasetMultipleGPUs(image_dir_train, mask_dir_train, transform=custom_transform, target_transform=custom_target_transform)#, data_augmentation=data_augmentation)
imagedataset_val = DeepGlobeDatasetMultipleGPUs(image_dir_val, mask_dir_val, transform=custom_transform, target_transform=custom_target_transform)#, data_augmentation=data_augmentation)
imagedataset_test = DeepGlobeDatasetMultipleGPUs(image_dir_test, mask_dir_test, transform=custom_transform, target_transform=custom_target_transform)#, data_augmentation=data_augmentation)

dataloader_train = DeepGlobeDatasetMultipleGPUs(imagedataset_train, batch_size=2, shuffle=True, num_workers=2)
dataloader_val = DeepGlobeDatasetMultipleGPUs(imagedataset_val, batch_size=2, shuffle=False, num_workers=2)
dataloader_test = DeepGlobeDatasetMultipleGPUs(imagedataset_test, batch_size=2, shuffle=False, num_workers=2)

resnet_unet = MultiGPU_ResNetUNet_with_comm(n_channels=3, resnet_type="resnet18", n_classes=7, input_shape=(1224, 1224), num_comm_fmaps=64, devices=["cuda:0"], depth=4, subdom_dist=(2, 2),
                 bilinear=False, comm=True, complexity=16, dropout_rate=0.1, kernel_size=5, padding=2, communicator_type=None)

total_params = sum(p.numel() for p in resnet_unet.encoders[0].parameters() if p.requires_grad)
print("Total number of parameters in encoders:", total_params)

def test_parallel_model(comm=True, plotting=True, num_epochs=50):

    unet = MultiGPU_ResNetUNet_with_comm(n_channels=3, n_classes=7, input_shape=(1224, 1224), num_comm_fmaps=64, devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"], depth=4, subdom_dist=(2, 2),
                 bilinear=False, comm=comm, complexity=64, dropout_rate=0.1, kernel_size=5, padding=2, communicator_type=None)
    unet.train()
    if comm:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters()) + list(unet.communication_network.parameters()) 
    else:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters())
        
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss = smp.losses.DiceLoss(mode="multiclass")
    losses = []

    # Wrap your training loop with tqdm
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_losses = []  # Initialize losses for the epoch
        for images, masks in tqdm(dataloader_train):
            images = [im.float() for im in images]
            masks = masks.to("cuda:0", dtype=torch.long)

            ## Forward propagation:
            # Run batch through encoder
            predictions = unet(images)
            
            ## Backward propagation
            l = loss(predictions, masks)
            l.backward()

            losses.append(l.item())  # Append loss to global losses list
            epoch_losses.append(l.item())  # Append loss to epoch losses list

            # Sum gradients of model2 to model1 (Allreduce)
#             with torch.no_grad():
#                 for i in range(1, len(unet.encoders)):
#                     for param1, param2 in zip(unet.encoders[0].parameters(), unet.encoders[i].parameters()):
#                         param1.grad += param2.grad.to("cuda:0")
#                         param2.grad = None

            with torch.no_grad():
                for i in range(1, len(unet.decoders)):
                    for param1, param2 in zip(unet.decoders[0].parameters(), unet.decoders[i].parameters()):
                        param1.grad += param2.grad.to("cuda:0")
                        param2.grad = None

            ## Parameter update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Update model2 with the state of model1
#             for i in range(1, len(unet.encoders)):
#                 unet.encoders[i].load_state_dict(unet.encoders[0].state_dict())
            for i in range(1, len(unet.decoders)):
                unet.decoders[i].load_state_dict(unet.decoders[0].state_dict())


        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print(f"Training the model {'with' if comm else 'without'} communication network took: {time.time() - start_time:.2f} seconds.")
    
    if plotting:
        unet.eval()
        preds = unet(images)
        
        for i in range(masks.shape[0]):
        
            pred = torch.argmax(preds, dim=1)[i].cpu()
            mask = masks.cpu()[i]

            plt.subplot(1,2,1)
            plt.title("Mask")
            plt.imshow(mask[0], cmap="gray")
            plt.axis("off")
            plt.subplot(1,2,2)
            plt.title("Prediction")
            plt.imshow(pred, cmap="gray")
            plt.axis("off")
            plt.show()

    return unet, losses


unet, losses = test_parallel_model(comm=True, plotting=True, num_epochs=50)
unet.save_weights(save_path="./figures/unet_without_comm.pth")
plt.plot(losses)
plt.savefig("loss_curve_with_comm.png", dpi=300, bbox_inches="tight")
plt.close()


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

            predicted_labels = model([image.float() for image in images])
            predicted_labels = torch.argmax(predicted_labels, dim=1)
            
            true_labels_list = true_labels.cpu().numpy().ravel()
            predicted_labels_list = predicted_labels.cpu().numpy().ravel()

            confusion += confusion_matrix(true_labels_list, predicted_labels_list, labels=range(num_classes))

            torch.cuda.empty_cache()

    return confusion

cf = evaluate_model_metrics(model=unet, dataloader=dataloader_test, device=None, num_classes=6)
from data_processing_functions import plot_heatmap_from_confusion, compute_iou_from_confusion
plot_heatmap_from_confusion(cf)
plt.savefig("./figures/confusion_matrix_with_comm.png", dpi=300, bbox_inches="tight")
compute_iou_from_confusion(cf)