import os
from .dataset_components import generate_image_and_mask

def generate_dataset(img_width, img_height = 32, num_images_train=4000, num_images_validation=2000, num_images_test=1000,
                     circle_radius = 4, num_circles = 2, line_width = 3, max_apart = 2, min_apart = 1, save_descr = ""):

    num_subdomains = img_width // 32
    image_size = (img_width, img_height)
    
    if save_descr == "":
        save_dir = f"./data/synthetic_data/synthetic_data_{num_subdomains}_subdomains"
    else:
        save_dir = f"./data/synthetic_data_{save_descr}/synthetic_data_{num_subdomains}_subdomains"
    
    img_dir = save_dir + "/images/"
    mask_dir = save_dir + "/masks/"

    img_dir_val = save_dir + "_validation/images/"
    mask_dir_val = save_dir + "_validation/masks/"

    img_dir_test = save_dir + "_test/images/"
    mask_dir_test = save_dir + "_test/masks/"

    # Create a directory to store the images and masks
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(img_dir_val, exist_ok=True)
    os.makedirs(mask_dir_val, exist_ok=True)
    os.makedirs(img_dir_test, exist_ok=True)
    os.makedirs(mask_dir_test, exist_ok=True)
    
    # Log the parameters used in the function
    log_file_path = save_dir + "/function_params.txt"
    params = {
        "img_width             ": img_width,
        "img_height            ": img_height,
        "img_size              ": image_size,
        "num_images_train      ": num_images_train,
        "num_images_validation ": num_images_validation,
        "num_subdomains        ": num_subdomains,
        "num_images_test       ": num_images_test,
        "circle_radius         ": circle_radius,
        "num_circles           ": num_circles,
        "line_width            ": line_width,
        "max_apart             ": max_apart,
        "min_apart             ": min_apart,
        "save_descr            ": save_descr,
        "save_dir              ": save_dir
    }

    with open(log_file_path, 'w') as log_file:
        for key, value in params.items():
            log_file.write(f"{key}: {value}\n")

    # Generate and save the images and masks
    for i in range(num_images_train):
        image, mask = generate_image_and_mask(image_size=image_size, num_circles=num_circles, 
                                              circle_radius=circle_radius, line_width=line_width, num_subdomains = num_subdomains,
                                              min_apart = min_apart, max_apart = max_apart, different_class=True)
        image.save(os.path.join(img_dir, f"image_{i}.png"))
        mask.save(os.path.join(mask_dir, f"mask_{i}.png"))
    

    # Generate and save the validation images and masks
    for i in range(num_images_validation):
        image, mask = generate_image_and_mask(image_size=image_size, num_circles=num_circles, 
                                              circle_radius=circle_radius, line_width=line_width, num_subdomains = num_subdomains,
                                              min_apart = min_apart, max_apart = max_apart, different_class=True)
        image.save(os.path.join(img_dir_val, f"image_{i}.png"))
        mask.save(os.path.join(mask_dir_val, f"mask_{i}.png"))

    # Generate and save the test images and masks
    for i in range(num_images_test):
        image, mask = generate_image_and_mask(image_size=image_size, num_circles=num_circles, 
                                              circle_radius=circle_radius, line_width=line_width, num_subdomains = num_subdomains,
                                              min_apart = min_apart, max_apart = max_apart, different_class=True)
        image.save(os.path.join(img_dir_test, f"image_{i}.png"))
        mask.save(os.path.join(mask_dir_test, f"mask_{i}.png"))

if __name__=="__main__":
    generate_dataset(img_width=64)