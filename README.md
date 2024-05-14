# SORTN (SAR-to-Optical Residual Translation Network)

Weights for SORTN model (required to run the code) can be downloaded from the following Google Drive link. 

https://drive.google.com/file/d/1OZqkzs6vYdwf3_y0_GiGy0Ycag6I1YPZ/view?usp=sharing

## Train
Command: 

```
pip3 install -r requirements.txt

python3 main.py <PATH TO DATASET>
```

Installs the required libraries from requirements.txt and runs the train script.

After running the code, it creates a Results directory where the generated images, losses and accuracies are stored.

Once training is completed, the latest SORTN checkpoint can be used to train SRUN (SAR Super-Resolution U-Net).
