from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from glob import glob
from tqdm import tqdm
import random
import argparse
import cv2
random.seed(10)
torch.manual_seed(10)

class SAR_optic_dataset(Dataset):
    def __init__(self, root_dir, tensor_transform = True):        
        self.tensor_transform = tensor_transform
        self.fetch_data_pairs(root_dir)
        if tensor_transform:
            self._init_transform() 

    def fetch_data_pairs(self, root_dir):
        sar_folder_names = sorted(glob(os.path.join(root_dir, '*', 's1_*')))
        self.sar_files = []
        self.optic_files = []
        print('Finding the paths of SAR and Optical Image files, this might take a minute...')
        for folder in sar_folder_names:  
            temp = sorted(glob(os.path.join(folder, '*')))
            for sar_path in temp:
                if random.random() < 0.25:  # Using only one-fourth of the dataset for training
                    # Finding the corresponding optic file path
                    parts = sar_path.split('_')
                    parts[-3] = 's2'
                    parts[1] = os.path.join(os.path.dirname(parts[1]), 's2')
                    optic_path = '_'.join(parts)
                    # Appending only if the corresponding optical images are present as well
                    if os.path.isfile(optic_path):
                        self.sar_files.append(sar_path)
                        self.optic_files.append(optic_path)
            
        
    def _init_transform(self):
        self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])


    def __getitem__ (self, index):
        optic_path, sar_path = self.optic_files[index], self.sar_files[index]
        sar_img = cv2.imread(sar_path)[:,:,0]
        optic_img = cv2.imread(optic_path)
        optic_img = cv2.cvtColor(optic_img, cv2.COLOR_BGR2RGB)

        if self.tensor_transform == True:
            optic_img = self.transform(optic_img)
            sar_img = self.transform(sar_img)

            if random.random() >= 0.5:       # Flipping along the first dimension
                optic_img = torch.flip(optic_img, [1])
                sar_img = torch.flip(sar_img, [1])
            if random.random() >= 0.5:       # Flipping along the second dimension
                optic_img = torch.flip(optic_img, [2])
                sar_img = torch.flip(sar_img, [2])
        
        return optic_img, sar_img

    def __len__ (self):
        return len(self.sar_files)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Dataset Initialization')
    parser.add_argument('--path', help='Path to the dataset')
    args = parser.parse_args()

    dataset = SAR_optic_dataset(args.path, tensor_transform=True)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
    x = iter(trainloader)
    o,s= x.next()
    o,s= x.next()
    print(o.shape, s.shape)
    # plt.imshow(o[0])
    plt.imshow(o[0].permute(1,2,0).detach())
    plt.savefig('trial1.png')
    plt.show()
    # plt.imshow(s[0], "gray")
    plt.imshow(s[0].permute(1,2,0).squeeze().detach(), "gray")
    plt.savefig('trial2.png')
    plt.show()