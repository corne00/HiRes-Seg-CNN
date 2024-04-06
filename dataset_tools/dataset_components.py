import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define transformations to apply to images and masks
def custom_transform(image):
    #image = image.resize((64, ))
    image = np.array(image)   # Convert to float and normalize
    image = image / np.max(image)
    #print(image.shape)
    
    image = np.expand_dims(image, axis=0)
    #image = np.transpose(image, (0, 1, 2))  # Change channel order to C x H x
    image = torch.from_numpy(image)
    return image

def custom_target_transform(mask):
    #mask = mask.resize((128, 128))
    mask = np.array(mask) / 255.0  # Convert to float and normalize
    #print(mask.shape)
    mask = np.expand_dims(mask, axis=0)  # Add channel dimension
    #mask = np.transpose(mask, (0, 1, 2))  # Change channel order to C x H x
    mask = torch.from_numpy(mask)
    return mask

def generate_image_and_mask(image_size, num_circles=1, circle_radius=4, line_width=2, num_subdomains=2, min_apart=1, max_apart=1, different_class=False):
    """
    This function generates one image with circles and the corresponding mask.

    Parameters:
    - image_size (tuple): The dimensions (width, height) of the output image.
    - num_circles (int): The number of circles to generate.
    - circle_radius (int): The radius of the circles.
    - line_width (int): The width of the lines connecting the circles in the mask.
    - num_subdomains (int): The number of subdomains to divide the image into.
    - min_apart (int): The minimum number of subdomains between circles (counting the one the circle is in too!). 
    - max_apart (int): The maximum number of subdomains between circles (counting the one the circle is in too!).
    - different_class (bool): A flag to determine if circles should belong to different classes(e.g., for segmentation tasks).

    Returns:
    - image (PIL.Image): The generated image with circles.
    - mask (PIL.Image): The corresponding mask image with lines connecting the circles.
    """
    
    # Calculate subdomain width and generate list with subdomain indices:
    subdomain_width = image_size[0] // num_subdomains
    subdomain_indices = list(range(num_subdomains))
    
    # Initialize images and masks
    image = Image.new("L", image_size, 0)
    mask = Image.new("L", image_size, 0)

    # Initialize drawing
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    
    for i in range(num_circles):
        # select the first subdomain where to draw randomly
        subdomain_1 = np.random.choice(subdomain_indices)

        # create a list of the possible subdomains for the other circles based on the function arguments
        min_subdomain2 = np.max([subdomain_1 - max_apart, 0])
        max_subdomain2 = np.min([subdomain_1 + max_apart, num_subdomains - 1])

        range_min = list(range(min_subdomain2, subdomain_1 + 1 - min_apart))
        range_max = list(range(subdomain_1 + min_apart, max_subdomain2 + 1))
        subdomain_2 = np.random.choice(range_min + range_max)

        # Define the subdomain range for the selected circle
        x1_range = (subdomain_1 * subdomain_width, (subdomain_1 + 1) * subdomain_width)
        x2_range = (subdomain_2 * subdomain_width, (subdomain_2 + 1) * subdomain_width)

        # Circle 1 centered in the left half of the image
        circle1_x = np.random.randint(x1_range[0] + circle_radius, x1_range[1] - circle_radius)
        circle1_y = np.random.randint(circle_radius, image_size[1] - circle_radius)
    
        # Circle 2 centered in the right half of the image
        circle2_x = np.random.randint(x2_range[0] + circle_radius, x2_range[1] - circle_radius)
        circle2_y = np.random.randint(circle_radius, image_size[1] - circle_radius)
    
        # Draw a line between the circles on the mask
        mask_draw.line([(circle1_x, circle1_y), (circle2_x, circle2_y)], fill=1+different_class*i*2, width=line_width)
        
        # Draw the two circles on both the image and the mask
        draw.ellipse([(circle1_x - circle_radius, circle1_y - circle_radius),
                      (circle1_x + circle_radius, circle1_y + circle_radius)], fill=different_class*i*2+2)
        mask_draw.ellipse([(circle1_x - circle_radius, circle1_y - circle_radius),
                      (circle1_x + circle_radius, circle1_y + circle_radius)], fill=different_class*i*2+2)
    
        draw.ellipse([(circle2_x - circle_radius, circle2_y - circle_radius),
                      (circle2_x + circle_radius, circle2_y + circle_radius)], fill=different_class*i*2+2)
        mask_draw.ellipse([(circle2_x - circle_radius, circle2_y - circle_radius),
                      (circle2_x + circle_radius, circle2_y + circle_radius)], fill=different_class*i*2+2)

    

    return image, mask

def plot_vlines(ax, image_size, num_subdomains):
    subdomain_length = image_size[0] // num_subdomains
    for i in range(num_subdomains-1):
        ax.vlines((i+1)*subdomain_length, 0, image_size[1]-1)


if __name__=="__main__":
    image, mask = generate_image_and_mask(image_size=(64,32), num_circles=1, circle_radius=4, line_width=2,
                                          num_subdomains=2, min_apart=1, max_apart=1, different_class=False)

    
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.suptitle("Example image and mask")
    plt.show()