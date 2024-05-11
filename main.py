from Models.SORTN import SORTN
from Models.SRUN_SinglePass import SRUN_SinglePass
from Models.SRUN_2Step import SRUN_2Step
from SAR_Optic_Dataset import SAR_optic_dataset, collate_func, SAR_optic_dataset_2Step_Val
import yaml
import argparse

from Runner.train import train
from Runner.validate import validate_singlepass, validate_2step

import torch
from torch import optim
from torch.utils.data import DataLoader
torch.manual_seed(10)

def load_weights(path,optimizer,scheduler,model):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    epoch = ckpt['epoch']
    return optimizer,scheduler,model,epoch

parser = argparse.ArgumentParser(description = 'calling for training or performing inference on the model')
parser.add_argument("dataset_path", help = "Dataset_path")
parser.add_argument("--last_checkpoint", default=None, help = "Provide checkpoint if continuing training")
parser.add_argument("--to_do", default = "train", help = "Valid Arguments: train, validate")
args = parser.parse_args()
if args.to_do != "train" and args.to_do != "validate":
    print('Please enter a valid argument. (train, validate)')
    exit()
to_do = args.to_do

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

if cfg['RESOLUTION_METHOD'] == 'Single Pass':
    srun_model = SRUN_SinglePass(scale_factor=cfg['SCALE_RATIO'], in_channels=1, filter_size=12, num_eram_layers=20)
elif cfg['RESOLUTION_METHOD'] == 'Two Step':
    srun_model = SRUN_2Step(scale_factor=4, in_channels=1, filter_size=12, num_eram_layers=20)
else:
    print("Non-valid Resolution Method mentioned in config Files. Valid ones are 'Single Pass' and 'Two Step'.")
    exit()

print('Model parameters: ', sum(p.numel() for p in srun_model.parameters()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
srun_model = srun_model.to(device)
generator = SORTN()
generator = generator.to(device)
generator.load_state_dict(torch.load(cfg['PRETRAINED_SORTN'])["generator"])

if to_do == 'train':
    
    print('Loading dataset for training')
    dataset = SAR_optic_dataset(cfg, args.dataset_path, tensor_transform=True)
    print(f'Dataset size: {len(dataset)} images')
    a = int(0.9 * len(dataset))
    b = len(dataset) - a
    train_ds, val_ds = torch.utils.data.random_split(dataset, (a, b))
    trainloader = DataLoader(train_ds, batch_size=64, collate_fn=collate_func, num_workers=8, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=64, collate_fn=collate_func, num_workers=8, shuffle=False)
    trainloader_plot = DataLoader(train_ds, batch_size=1, collate_fn=collate_func, num_workers=8, shuffle=True)
    valloader_plot = DataLoader(val_ds, batch_size=1, collate_fn=collate_func, num_workers=8, shuffle=False)
    
    if args.last_checkpoint is not None:
        
        initial_lr = cfg['TRAIN']['INITIAL_LR']
        beta_1 = cfg['TRAIN']['BETA_1']
        beta_2 = cfg['TRAIN']['BETA_2']
        optimizer = optim.Adam(srun_model.parameters(),lr=initial_lr,betas=(beta_1,beta_2))
        decay_factor = cfg['TRAIN']['DECAY_FACTOR']
        decay_patience = cfg['TRAIN']['PATIENCE']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=decay_factor,patience=decay_patience,verbose=True)
        
        print('Loading pretrained weights for training')
        optimizer,scheduler,srun_model,last_epoch = load_weights(args.last_checkpoint,optimizer,scheduler,srun_model)
        start_epoch = last_epoch + 1
        print('Starting training for epoch ', start_epoch)
        
    else:
        print('Training the model from scratch')
        start_epoch = 0
        
        initial_lr = cfg['TRAIN']['INITIAL_LR']
        beta_1 = cfg['TRAIN']['BETA_1']
        beta_2 = cfg['TRAIN']['BETA_2']
        optimizer = optim.Adam(srun_model.parameters(),lr=initial_lr,betas=(beta_1,beta_2))
        decay_factor = cfg['TRAIN']['DECAY_FACTOR']
        decay_patience = cfg['TRAIN']['PATIENCE']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=decay_factor,patience=decay_patience,verbose=True)
        
    train(cfg, trainloader, valloader, trainloader_plot, valloader_plot, srun_model, generator, optimizer, scheduler, start_epoch)
    
elif to_do == 'validate':
    print('Loading dataset for validation')
    if cfg['RESOLUTION_METHOD'] == 'Single Pass':
        dataset = SAR_optic_dataset(cfg, args.dataset_path, tensor_transform=True)
    else:
        dataset = SAR_optic_dataset_2Step_Val(cfg, args.dataset_path, tensor_transform=True)
    infer_loader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=False)
    weights = torch.load(cfg['SRUN_VALIDATION_CKPT'])
    srun_model.load_state_dict(weights['model'])

    if cfg['RESOLUTION_METHOD'] == 'Single Pass':
        validate_singlepass(cfg, infer_loader, srun_model, generator)
    else:
        validate_2step(cfg, infer_loader, srun_model, generator)