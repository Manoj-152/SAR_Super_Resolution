from Utils.plotters import train_plotter, test_plotter, plot_feature_maps
from Utils.ssim import ssim
from Utils.Performance_Measures import psnr

from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(10)

# Using linear learning rate decay scheduler
def linear_lr_decay_scheduler(optimizer, epoch, init_lr = 4e-4, decay_after_epoch = 50):
    if epoch > decay_after_epoch:
        lr = init_lr - (init_lr / 100.01)*(epoch - decay_after_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer

# Function to save weights of generator and discriminator together
def save_weights(path,generator,discriminator,epoch):
    torch.save({
        "discriminator": discriminator.state_dict(),
        "generator": generator.state_dict(),
        "epoch": epoch
    }, path)


def train(trainloader, valloader, generator, discriminator, optimizer_gen, optimizer_disc, start_epoch):
    os.makedirs('Results/checkpoints', exist_ok=True)
    os.makedirs('Results/Generated_Images', exist_ok=True)
    os.makedirs(os.path.join('Results/Generated_Images', 'Generator_Train_Results'), exist_ok=True)
    os.makedirs(os.path.join('Results/Generated_Images', 'Generator_Test_Results'), exist_ok=True)
    os.makedirs('Results/Losses', exist_ok=True)
    os.makedirs('Results/Accuracies', exist_ok=True)
    
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    
    saving_after_epochs = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
        
    generator = generator.to(device)
    discriminator = discriminator.to(device)
        
    for epoch in range(start_epoch, num_epochs):
        # Starting training
        running_G_loss = 0.
        running_D_loss = 0.
        running_l1_loss = 0.
        running_ssim = 0.
        running_psnr = 0.
        
        optimizer_gen = linear_lr_decay_scheduler(optimizer_gen, epoch+1, init_lr=0.0004)
        optimizer_disc = linear_lr_decay_scheduler(optimizer_disc, epoch+1, init_lr=0.0004)
        
        for n,(optic,sar) in enumerate(tqdm(trainloader)):
            if n == len(trainloader) - 1: break
    
            optic = optic.to(device)
            sar = sar.to(device)
    
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()
            batch_size = optic.size(0)
    
            # Training the generator first
            # maximize log(D(G(x))) - alpha*l1_loss(y,G(x))
            label = torch.full((batch_size, ), real_label, dtype=torch.float)
            label = label.to(device)
    
            alpha = 100
            optic_gen,_ = generator(sar)
            pred_fake,_ = discriminator(optic_gen)
            l1_loss = F.l1_loss(optic_gen, optic)
            errG = criterion(pred_fake.flatten(), label) + alpha*l1_loss
            errG.backward()
            optimizer_gen.step()
    
            # Training the discriminator
            # maximize log(D(y)) + log(1 - D(G(x)))
            label.fill_(real_label)
            pred_real,_ = discriminator(optic)
            errD_real = criterion(pred_real.flatten(), label)
            errD_real.backward()
            
            label.fill_(fake_label)
            optic_gen,_ = generator(sar)
            pred_fake,_ = discriminator(optic_gen)
            errD_fake = criterion(pred_fake.flatten(), label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizer_disc.step()
            
            running_G_loss += errG.item()
            running_D_loss += errD.item()
            running_l1_loss += l1_loss.item()
            
            ssim_score = ssim(optic_gen, optic, val_range=2, minmax=True)
            psnr_score = psnr(optic_gen, optic, minmax=True)
            running_ssim += ssim_score.item()
            running_psnr += psnr_score.item()
        
        with open('Results/Losses' + '/gen_overall_train_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_G_loss / (len(trainloader) - 1)))
        with open('Results/Losses' + '/disc_overall_train_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_D_loss / (len(trainloader) - 1)))
        with open('Results/Losses' + '/l1_overall_train_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_l1_loss / (len(trainloader) - 1)))
        with open('Results/Accuracies' + '/ssim_score_train.txt', 'a') as f:
            f.write("%s\n" % str(running_ssim / (len(trainloader) - 1)))
        with open('Results/Accuracies' + '/psnr_score_train.txt', 'a') as f:
            f.write("%s\n" % str(running_psnr / (len(trainloader) - 1)))
    
        print('Epoch: ',epoch,' Gen_Train_loss: ',round(running_G_loss / (len(trainloader) - 1), 4), \
                    ' Disc_Train_loss: ',round(running_D_loss / (len(trainloader) - 1), 4),  ' SSIM score: ',round(running_ssim / (len(trainloader) - 1), 4), \
                            ' PSNR score: ',round(running_psnr / (len(trainloader) - 1), 4))
    
    
        # Saving weights once every 'saving_after_epochs' epochs
        if epoch%saving_after_epochs == 0 or epoch == num_epochs - 1:
            save_weights('Results/checkpoints' + "/Epoch"+str(epoch)+".pth", generator, discriminator, epoch)
        
        
        # Starting validation
        running_G_loss = 0.
        running_D_loss = 0.
        running_l1_loss = 0.
        running_ssim = 0.
        running_psnr = 0.
        
        for n,(optic,sar) in enumerate(tqdm(valloader)):
            #if n == 1: break
            if n == len(valloader) - 1: break
            
            optic = optic.to(device)
            sar = sar.to(device)
            batch_size = optic.size(0)
    
            # Training the generator first
            # maximize log(D(G(x))) - alpha*l1_loss(y,G(x))
            label = torch.full((batch_size, ), real_label, dtype=torch.float)
            label = label.to(device)
    
            alpha = 100
            with torch.no_grad():
                optic_gen,_ = generator(sar)
                pred_fake,_ = discriminator(optic_gen)
            l1_loss = F.l1_loss(optic_gen, optic)
            errG = criterion(pred_fake.flatten(), label) + alpha*l1_loss
    
            # Training the discriminator
            # maximize log(D(y)) + log(1 - D(G(x)))
            label.fill_(real_label)
            with torch.no_grad():
                pred_real,_ = discriminator(optic)
            errD_real = criterion(pred_real.flatten(), label)
            
            label.fill_(fake_label)
            with torch.no_grad():
                optic_gen,_ = generator(sar)
                pred_fake,_ = discriminator(optic_gen)
            errD_fake = criterion(pred_fake.flatten(), label)
            errD = errD_real + errD_fake
            
            running_G_loss += errG.item()
            running_D_loss += errD.item()
            running_l1_loss += l1_loss.item()
            
            ssim_score = ssim(optic_gen, optic, val_range=2, minmax=True)
            psnr_score= psnr(optic_gen, optic, minmax=True)
            running_ssim += ssim_score.item()
            running_psnr += psnr_score.item()

        with open('Results/Losses' + '/gen_overall_test_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_G_loss / (len(valloader) - 1)))
        with open('Results/Losses' + '/disc_overall_test_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_D_loss / (len(valloader) - 1)))
        with open('Results/Losses' + '/l1_overall_test_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_l1_loss / (len(valloader) - 1)))
        with open('Results/Accuracies' + '/ssim_score_test.txt', 'a') as f:
            f.write("%s\n" % str(running_ssim / (len(valloader) - 1)))
        with open('Results/Accuracies' + '/psnr_score_test.txt', 'a') as f:
            f.write("%s\n" % str(running_psnr / (len(valloader) - 1)))
    
        print('Epoch: ',epoch,' Gen_Test_loss: ',round(running_G_loss / (len(valloader) - 1), 4), \
                ' Disc_Test_loss: ',round(running_D_loss / (len(valloader) - 1), 4), ' SSIM score: ',round(running_ssim / (len(valloader) - 1), 4), \
                       ' PSNR score: ',round(running_psnr / (len(valloader) - 1), 4))
                      
        # plotting generated outputs and feature maps
        train_plotter(device, generator, trainloader, epoch, 'Results/Generated_Images' + '/Generator_Train_Results/')
        test_plotter(device, generator, valloader, epoch, 'Results/Generated_Images' + '/Generator_Test_Results/')
    
        # if epoch%20 == 0: 
        #     plot_feature_maps(device, generator, discriminator, trainloader, epoch, 'Results/Generated_Images' + '/')