# SAR-Super-resolution-v1

This repository contains a PyTorch implementation of **OGSRN** (Optical-Guided Super-Resolution Network), a framework designed to reconstruct high-resolution Synthetic Aperture Radar (SAR) images by leveraging optical imagery for guidance. The architecture employs a two-stage approach to overcome speckle noise and detail loss: first, the **SRUN** (SAR Super-Resolution U-Net) performs the initial super-resolution reconstruction to focus on high-frequency details; second, the **SORTN** (SAR-to-Optical Residual Translation Network) acts as a guidance mechanism, translating the reconstructed SAR output into the optical domain to ensure structural consistency and constrain the solution space using clean optical features.

<p align="center">
  <img src="Assets/model_architecture.png" width="400">
</p>

## Model Weights

Weights for SORTN model (required to run the code) can be downloaded from the following google drive link. 

https://drive.google.com/file/d/1OZqkzs6vYdwf3_y0_GiGy0Ycag6I1YPZ/view?usp=drive_link

The SRUN (Super-Resolution Model) checkpoints are available in the Best Checkpoints folder. They are available for 2x, 4x, 8x, 16x, and 2-Step Resolution Process.

## Dataset

Please download the [Sentinel 1-2](https://mediatum.ub.tum.de/1436631) dataset (There are 4 folders with SAR-Optic Images pairs of around 40GB). Mention the dataset path later while running the main.py code for training and validation.

## Train
Command: 

```
pip3 install -r requirements.txt

python3 main.py <PATH TO DATASET>
```

Installs the required libraries from requirements.txt and runs the train script.

Before running the code, respective directories for the dataset and the directories where the results are to be stored must be given in the config.yaml file. New folders will be created for result directories already if they are not present.

The resolution method (Single Pass and Two Step) can be changed in the config file along with the resolution scales.

## Validation
Command:

```
pip3 install -r requirements.txt

python3 main.py <PATH TO DATASET> --to_do validate
```

The validation code outputs the validation accuracy obtained along with storing 20 random validation results.

## Inference main jupyter notebook
This notebook contains the same codes which are used for inference in the script. It provides a visual understanding of what is done at each step (from preprocessing to super-resolving the image).

## Solution Development
* The model is trained on 51k SAR-optic pairs of patches. Patch size is 256x256
* Model: SORTN (Present in next branch) is the generator using to obtain optic image from SAR
* SRUN: Responsible for super-resolving the image, architecture similar to U-Net
* Loss Functions: Content Loss + 0.1 x Evaluation Loss is used for SRUN. cGAN loss + 100 * L1 loss between optical ground truth and optical generated is used for SORTN.
* For inference, the inference input is cropped into required patch sizes. The patches are super-resolved and later stitched together. The steps can be visualized in the Inference_main jupyter notebook.

## Results
(a.) Spacenet 6 dataset
![Screen Shot 1944-02-23 at 3 05 57 PM](https://user-images.githubusercontent.com/82506345/168256472-910eadd5-8345-4a6c-8bb4-84dfb5758c45.png)



(b.) Sentinel -1 

![imgonline-com-ua-twotoone-RgxpMML7YNeRf](https://user-images.githubusercontent.com/82506345/168259244-e30333f6-6dff-4788-891d-23eff516af76.jpeg)


