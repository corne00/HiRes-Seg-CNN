# HiRes-Seg-CNN

**Domain Decomposition-Inspired CNN for Image Segmentation: Leveraging Multiple GPUs for High-Resolution Images**

Welcome to the repository for the thesis titled “A Domain Decomposition-based CNN Architecture for High-Resolution Image Segmentation.” This repository contains the code scripts used to generate the results presented in the thesis. Below is a brief overview of the repository structure:

### Folders and files:

- `data`: This folder is dedicated to storing synthetic and realistic datasets.
- `dataset_tools`: Contains several useful scripts for initializing datasets, dataloaders, data transformations, processing data, and functions to generate synthetic datasets.
- `models_2d`: This directory holds Torch scripts initializing the model proposed in the thesis, along with a baseline U-Net model and various model components such as the encoder, decoder, and communication network.
- `visualize_results.py`: This file houses several functions to help visualize results, including plotting heat functions and reading log scripts.
- `evaluate.py`: Contains functions used to evaluate model metrics by generating a confusion matrix.
- `train_new_approach.py`: Holds the function used to train a model for segmentation.

### How to Use:

1. **Data Preparation**: Place your datasets in the `data` folder or use the provided scripts in `dataset_tools` to generate synthetic datasets.
2. **Model Initialization**: Access the scripts in the `models_2d` folder to initialize the proposed model or baseline models.
3. **Training**: Utilize `train_new_approach.py` to train your segmentation model.
4. **Evaluation**: After training, use `evaluate.py` to evaluate model performance and generate confusion matrices.
5. **Visualization**: Visualize results using the functions provided in `visualize_results.py`.
