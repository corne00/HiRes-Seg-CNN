**Structure of the code** (https://github.com/corne00/HiRes-Seg-CNN)

- `DataLoader`: [HiRes-Seg-CNN/dataset_tools/DataSets_multiple_GPUs.py at main · corne00/HiRes-Seg-CNN (github.com)](https://github.com/corne00/HiRes-Seg-CNN/blob/main/dataset_tools/DataSets_multiple_GPUs.py#L8)
  - Loads images, splits them along the height and width into $m \times n$ non overlapping subdomains. Applies data augmentation (optionally) and returns a list of sub-images and the *full* mask.
- `DDU-Net`: [HiRes-Seg-CNN/models_multiple_GPUs/MultiGPU_UNet_with_comm.py at main · corne00/HiRes-Seg-CNN (github.com)](https://github.com/corne00/HiRes-Seg-CNN/blob/main/models_multiple_GPUs/MultiGPU_UNet_with_comm.py)
  - Initializes an encoder + decoder clone. Contains several helper function to get the correct device, convert grid index to list index and vice versa. The input sub-images are partitioned across the available devices. First, encoders are processing the sub-images, then communication is performed, then decoding. 
    - Seems sequential but happens for the largest part in parallel due to the asynchronous working of PyTorch.
  - Prediction masks are concatenated to form one global mask!
- `training`: [HiRes-Seg-CNN/train_deepglobe_resnet_unet.py at main · corne00/HiRes-Seg-CNN (github.com)](https://github.com/corne00/HiRes-Seg-CNN/blob/main/train_deepglobe_resnet_unet.py) 
  - In our code, we use mixed precision training to increase training speed.
  - The communication network is stored on GPU:0 (or the first device in the list of available devices), might be hardcoded somewhere. Therefore, the masks need to be sent to this GPU after loading the data.
  - **Training procedure**
    1. *Forward propagation*: Note that for proper backward propagation, the communicated feature maps need to be cloned to prevent overwriting values (that are necessary for backward propagation)
    2. *Loss computation*: The training loss is computed for the **global** mask and **global** predictions. Could also happen locally per GPU, but this leads to to problems with the computation graph (in the current implementation). 
    3. *Backward propagation*: PyTorch itself handles the backward propagation through the entire network
    4. *Gradient accumulation*: The gradients of the encoders a decoders are aggregated on GPU:0. Aggregation is not necessary for the communication network (!).
    5. *Optimizer step*: update the weights based on the gradients, only for the encoder and decoder on GPU:0 + the communication network (other GPUs are not necessary!)
    6. *Communicate weights*: Share the weights of the encoder + decoder on GPU:0 with the other devices

<img src="https://lh7-us.googleusercontent.com/slidesz/AGV_vUcBKp7YR_HCT0hCOPfsrZ7LK1e1VXK3d8psRcm_3RuzNgyDsRnLg4W9cRIjXemFCVZroPYL4QBCLSl3JhxsWG5SSiw0WrKxKEw2zKZ8puOUcrDfleUr8lpP_PZadW7kwrObo-E0BHBcHUybaeMzeQLtiCXkd1m5=s2048?key=NASscbP7VBvDUr1arAXJQw" alt="img" width="1000">

<img src="https://lh7-us.googleusercontent.com/slidesz/AGV_vUc5fBvvB9YNdinHVgjnUv0ppZSvpWw5IOLIozygJ8tZd82bYYk2qY6SPKIsMxGu9D7uwGUjW2xoEXsQEdiqCoupOifYGm6Wnt1Jkst-2xJjo3iAPnu5fM5Ch_BDLwxmNaiAqoXmGp2_1vpHLUCxK8roQfPMUIUZ=s2048?key=NASscbP7VBvDUr1arAXJQw" alt="img" width="1000" />

