from Models.SORTN import SORTN
from Models.PatchGAN import PatchGAN
from SAR_Optic_Dataset import SAR_optic_dataset
import yaml
import argparse

from train import train

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
torch.manual_seed(10)


parser = argparse.ArgumentParser(description = 'calling for training or performing inference on the model')
parser.add_argument("dataset_path", help = "Dataset_path")
parser.add_argument("--last_checkpoint", default=None, help = "Provide checkpoint if continuing training")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = SORTN()
discriminator = PatchGAN(3)
generator = generator.to(device)
discriminator = discriminator.to(device)

# loading pretrained models
    
print('Loading dataset for training')
dataset = SAR_optic_dataset(args.dataset_path, tensor_transform=True)
print(f'Dataset size: {len(dataset)} images')
a = int(0.9 * len(dataset))
b = len(dataset) - a
train_ds, val_ds = torch.utils.data.random_split(dataset, (a, b))
trainloader = DataLoader(train_ds, batch_size=128, num_workers=8, shuffle=True, pin_memory=True)
valloader = DataLoader(val_ds, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)

if args.last_checkpoint is not None:
    print('Loading pretrained weights for training')
    weights = torch.load(args.last_checkpoint)
    generator.load_state_dict(weights['generator'])
    discriminator.load_state_dict(weights['discriminator'])
    start_epoch = weights['epoch'] + 1

    print('Starting training for epoch ', start_epoch)
else:
    print('Training the model from scratch')
    start_epoch = 0
    
# initializing optimizers
initial_lr = 0.0004
beta_1 = 0.5
beta_2 = 0.99
optimizer_gen = optim.Adam(generator.parameters(),lr=initial_lr,betas=(beta_1,beta_2))
optimizer_disc = optim.Adam(discriminator.parameters(),lr=initial_lr,betas=(beta_1,beta_2)) 
train(trainloader, valloader, generator, discriminator, optimizer_gen, optimizer_disc, start_epoch)