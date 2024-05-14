# SAR-Super-resolution-v1

Weights for SORTN model (required to run the code) can be downloaded from the following google drive link. 

https://drive.google.com/file/d/1OZqkzs6vYdwf3_y0_GiGy0Ycag6I1YPZ/view?usp=drive_link

## Train
Command: 

```
pip3 install -r requirements.txt

python3 main.py <PATH TO DATASET>
```

Installs the required libraries from requirements.txt and runs the train script.

Before running the code, respective directories for the dataset and the directories where the results are to be stored must be given in the config.yaml file. New folders will be created for result directories already if they are not present.

The resolution method (Single Pass and Two Step) can be changed in the config file along with the resolution scales.

## Test/Inference
Command:

```
pip3 install -r requirements.txt

python3 main.py <PATH TO DATASET> --to_do validate
```

The validation code outputs the validation accuracy obtained along with storing 20 random validation results.

## Test_Sentinel jupyter notebook
This notebook contains the same codes which are used for inference in the script. It provides a visual understanding of what is done at each step (from preprocessing to super-resolving the image).

## Solution Development
* The model is trained on 51k SAR-optic pairs of patches. Patch size is 256x256
* Model: SORTN (Present in next branch) is the generator using to obtain optic image from SAR
* SRUN: Responsible for super-resolving the image, architecture similar to U-Net
* Loss Functions: Content Loss + 0.1 x Evaluation Loss is used for SRUN. cGAN loss + 100 * L1 loss between optical ground truth and optical generated is used for SORTN.
* For inference, the inference input is cropped into required patch sizes. The patches are super-resolved and later stitched together. The steps can be visualized in the Test_Sentinel jupyter notebook.

## Results
(a.) Spacenet 6 dataset
![Screen Shot 1944-02-23 at 3 05 57 PM](https://user-images.githubusercontent.com/82506345/168256472-910eadd5-8345-4a6c-8bb4-84dfb5758c45.png)



(b.) Sentinel -1 

![imgonline-com-ua-twotoone-RgxpMML7YNeRf](https://user-images.githubusercontent.com/82506345/168259244-e30333f6-6dff-4788-891d-23eff516af76.jpeg)


