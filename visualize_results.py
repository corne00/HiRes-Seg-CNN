import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_results_deepglobe(dataset, models, model_names=None, num_examples=10):
    image, mask = dataset[0]

    DOMAIN_SIZE = (image.shape[2], image.shape[1])
    N_X = DOMAIN_SIZE[0] // 256
    N_Y = DOMAIN_SIZE[1] // 256
    NUM_EXAMPLES = num_examples
    
    torch.cuda.empty_cache()

    def plot_vhlines(ax, image_size, n_x, n_y):
        subdomain_length = image_size[0] // n_x
        subdomain_height = image_size[1] // n_y
        for i in range(n_x-1):
            ax.vlines((i+1)*subdomain_length-0.5, 0, image_size[1]-1)
        for i in range(n_y):
            ax.hlines((i+1)*subdomain_height-0.5, 0, image_size[1]-1)

    def denormalize_tensor(image, mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
        return image

    def compute_metrics(pred, mask, num_classes=6):
        ious = [0] * num_classes  # Initialize list using multiplication

        for i in range(num_classes):
            # Use torch.logical_and for element-wise logical AND
            intersection = torch.sum(torch.logical_and(pred == i, mask == i))
            union = torch.sum(torch.logical_or(pred == i, mask == i))

            # Use torch.div for element-wise division
            ious[i] = round(float(intersection / union), 3) if union != 0 else 0

        # Convert accuracy tensor to Python float before rounding
        accuracy = float(torch.sum(pred == mask).float() / pred.numel())

        return round(accuracy, 3), ious

    def labelmap_to_colormap(pred):
        label_map_colors = {
            0: [0, 255, 255],
            1: [255, 255, 0],
            2: [255, 0, 255],
            3: [0, 255, 0],
            4: [0, 0, 255],
            5: [255, 255, 255],
            6: [0, 0, 0]
        }
        labelmap = pred.squeeze()
        colormap = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)

        for label, color in label_map_colors.items():
            colormap[labelmap == label] = color
        return torch.tensor(colormap).permute(2, 0, 1)


    for model in models:
        model.eval()

    for i in range(NUM_EXAMPLES):
        plt.figure(figsize=(7.5*(len(models)+2), 7.5))
        image, mask = dataset = [np.random.randint(0, len(dataset))]

        preds = []
        metrics = []

        with torch.no_grad():
            for model in models:
                pred = model(image.to(device, dtype=torch.float).unsqueeze(0)).cpu()
                metrics = compute_metrics(pred, dim=1)
                pred = labelmap_to_colormap(pred)

                preds.append(pred)
                metrics.append(metric)

    plot_width = len(models) + 2

    for model_id in range(len(models)):
        ax = plt.subplot(1,plot_width, model_id+1)
        plt.title(model_names[model_id])
        plt.imshow(preds[model_id].permute(1,2,0))
        plot_vhlines(ax, DOMAIN_SIZE, N_X, N_Y)
        plt.axis("off")

    ax = plt.subplot(1,plot_width,plot_width-1)
    plt.title("True mask")
    plt.imshow(labelmap_to_colormap(mask).permute(1,2,0), vmin=0, vmax=6)
    plot_vhlines(ax, DOMAIN_SIZE, N_X, N_Y)
    plt.axis("off")

    ax = plt.subplot(1,plot_width,plot_width-1)
    plt.title("Image")
    plt.imshow(denormalize_tensor(image).permute(1,2,0))
    plt.imshow(labelmap_to_colormap(mask).permute(1,2,0), alpha=0.0003)
    plot_vhlines(ax, DOMAIN_SIZE, N_X, N_Y)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    