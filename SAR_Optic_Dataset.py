import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import argparse

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
torch.manual_seed(10)
import random
random.seed(10)


def collate_func(batch):
    temp_random = random.random()
    optic_imgs = []
    tmc_lr_imgs = []
    tmc_hr_imgs = []
    res_labels = []
    for b in batch:
        cfg, optic_img, tmc_img = b
        if cfg['RESOLUTION_METHOD'] == 'Single Pass':
            scale = cfg['SCALE_RATIO']
            width1 = int(tmc_img.shape[1]/scale)
            height1 = int(tmc_img.shape[0]/scale)
            dim1 = (width1, height1)
            width2 = int(tmc_img.shape[1]/1)
            height2 = int(tmc_img.shape[0]/1)
            dim2 = (width2, height2)
            tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
            tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
            res_label = 0

        elif cfg['RESOLUTION_METHOD'] == 'Two Step':
            if temp_random <= 0.33:
                width1 = int(tmc_img.shape[1]/16)
                height1 = int(tmc_img.shape[0]/16)
                dim1 = (width1, height1)
                width2 = int(tmc_img.shape[1]/4)
                height2 = int(tmc_img.shape[0]/4)
                dim2 = (width2, height2)
                tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
                tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
                res_label = 1
            elif temp_random <= 0.67:
                width1 = int(tmc_img.shape[1]/8)
                height1 = int(tmc_img.shape[0]/8)
                dim1 = (width1, height1)
                width2 = int(tmc_img.shape[1]/2)
                height2 = int(tmc_img.shape[0]/2)
                dim2 = (width2, height2)
                tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
                tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
                res_label = 2
            else:
                width1 = int(tmc_img.shape[1]/4)
                height1 = int(tmc_img.shape[0]/4)
                dim1 = (width1, height1)
                width2 = int(tmc_img.shape[1]/1)
                height2 = int(tmc_img.shape[0]/1)
                dim2 = (width2, height2)
                tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
                tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
                res_label = 4

        else:
            print("Non-valid Resolution Method mentioned in config Files. Valid ones are 'Single Pass' and 'Two Step'.")
            exit()
        
        tmc_img_hr = Image.fromarray(tmc_img_hr)
        tmc_img_lr = Image.fromarray(tmc_img_lr)

        transform_list = []
        if random.random() >= 0.5:
            transform_list.append(transforms.RandomVerticalFlip(1))
        if random.random() >= 0.5:
            transform_list.append(transforms.RandomHorizontalFlip(1))
        transform_list.append(transforms.ToTensor())
        transform_img = transforms.Compose(transform_list)
        
        optic_img = transform_img(optic_img)
        tmc_img_lr = transform_img(tmc_img_lr)
        tmc_img_hr = transform_img(tmc_img_hr)

        optic_imgs.append(optic_img)
        tmc_hr_imgs.append(tmc_img_hr)
        tmc_lr_imgs.append(tmc_img_lr)
        res_labels.append(res_label)

    return torch.stack(optic_imgs), torch.stack(tmc_hr_imgs), torch.stack(tmc_lr_imgs), torch.tensor(res_labels).unsqueeze(1)



class SAR_optic_dataset(Dataset):
    def __init__(self, cfg, root_dir, tensor_transform = True):
        self.tensor_transform = tensor_transform
        self.fetch_data_pairs(root_dir)
        self.cfg = cfg

    def fetch_data_pairs(self, root_dir):
        sar_folder_names = sorted(glob(os.path.join(root_dir, '*', 's1_*')))
        self.sar_files = []
        self.optic_files = []
        print('Finding the paths of SAR and Optical Image files, this might take a minute...')
        for folder in sar_folder_names:  
            temp = sorted(glob(os.path.join(folder, '*')))
            for sar_path in temp:
                if random.random() < 0.20:  # Using only one-fourth of the dataset for training
                    # Finding the corresponding optic file path
                    parts = sar_path.split('_')
                    parts[-3] = 's2'
                    parts[1] = os.path.join(os.path.dirname(parts[1]), 's2')
                    optic_path = '_'.join(parts)
                    # Appending only if the corresponding optical images are present as well
                    if os.path.isfile(optic_path):
                        self.sar_files.append(sar_path)
                        self.optic_files.append(optic_path)
                        # break


    def __getitem__ (self, index):
        optic_path, sar_path = self.optic_files[index], self.sar_files[index]
        optic_img = Image.open(optic_path).convert('RGB')
        sar_img = cv2.imread(sar_path)[:,:,0]
        
        return self.cfg, optic_img, sar_img

    def __len__ (self):
        return len(self.sar_files)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description='Dataset Initialization')
    parser.add_argument('dataset_path', help='Path to the dataset')
    args = parser.parse_args()

    dataset = SAR_optic_dataset(cfg, args.dataset_path, tensor_transform=True)
    trainloader = DataLoader(dataset, batch_size=1, collate_fn=collate_func, shuffle=True)
    x = iter(trainloader)
    o,s_hr,s_lr,res_label = x.next()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(o[0].permute(1,2,0).squeeze().detach())
    ax[1].imshow(s_hr[0].permute(1,2,0).squeeze().detach(), "gray")
    ax[2].imshow(s_lr[0].permute(1,2,0).squeeze().detach(), "gray")
    plt.savefig('trial.png',dpi=150)
    plt.show()