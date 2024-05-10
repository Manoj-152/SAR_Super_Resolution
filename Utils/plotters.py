import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import random
import os

# Function to plot generated images on trainloader
def train_plotter(device, generator, trainloader, epoch, path):
    batch_tensor = None
    print('Plotting the Generator Train results')
    for n,(optic,sar) in enumerate(trainloader):
        if n == 10: break
        optic = optic.to(device)
        sar = sar.to(device)
        
        with torch.no_grad():
            optic_gen,_ = generator(sar)
            optic_gen = optic_gen[0].unsqueeze(0)
        if batch_tensor is None:
            batch_tensor = optic_gen
        else:
            batch_tensor = torch.cat([batch_tensor, optic_gen], dim=0)
        optic = optic[0].unsqueeze(0)
        batch_tensor = torch.cat([batch_tensor, optic], dim=0)

    grid_img = vutils.make_grid(batch_tensor, nrow=4)
    plt.imshow(grid_img.permute(1,2,0).cpu().detach().numpy())
    plt.savefig(path + 'Generated_Image_epoch'+str(epoch)+'.png', dpi=350)

# Function to plot generated images on valloader
def test_plotter(device, generator, valloader, epoch, path):
    batch_tensor = None
    print('Plotting the Generator Test results')
    number_cnt = 0
    for n,(optic,sar) in enumerate(valloader):
        if number_cnt >= 10: break
        if random.random() > 0.98:
            optic = optic.to(device)
            sar = sar.to(device)
            
            with torch.no_grad():
                optic_gen,_ = generator(sar)
            if batch_tensor is None:
                batch_tensor = optic_gen
            else:
                batch_tensor = torch.cat([batch_tensor, optic_gen], dim=0)
            batch_tensor = torch.cat([batch_tensor, optic], dim=0)
            number_cnt += 1

    grid_img = vutils.make_grid(batch_tensor, nrow=4)
    plt.imshow(grid_img.permute(1,2,0).cpu().detach().numpy())
    plt.savefig(path + 'Generated_Image_epoch'+str(epoch)+'.png', dpi=350)
    

def plot_features(save_dir, features, tag):
    for i in range(len(features)):
        if features[i].size(1) >= 10:  temp = features[i][0][10].cpu().detach()
        else: temp = features[i][0][0].cpu().detach()
        plt.imshow(temp, "gray")
        plt.savefig(save_dir + '/Filter_'+str(i+1) + '_'+tag+'.png', dpi=150)
        
# Function to plot feature maps (uses the above function)
def plot_feature_maps(device, generator, discriminator, trainloader, epoch, path):
    print('Plotting the features maps')
    for n,(optic,sar) in enumerate(trainloader):
        if n == 5: break
        optic = optic.to(device)
        sar = sar.to(device)
        
        with torch.no_grad():
            _, fea_maps_gen = generator(sar)
            _, fea_maps_disc = discriminator(optic)
        
        save_dir = path + 'Generated_Features_epoch'+str(epoch)+'_picture'+str(n)
        os.makedirs(save_dir, exist_ok=True)
        plt.imshow(optic[0][0].cpu().detach(), "gray")
        plt.savefig(save_dir + '/Optic_Img.png', dpi=350)

        plot_features(save_dir, fea_maps_gen, 'gen')
        plot_features(save_dir, fea_maps_disc, 'disc')