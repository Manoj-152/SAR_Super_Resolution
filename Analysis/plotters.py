import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import random
import os
import torch.nn.functional as F
import cv2


def train_plotter(cfg, srun_model, generator, trainloader, epoch, scale_factor, path):
    batch_tensor1 = None
    batch_tensor2 = None
    print('Plotting the SRUN Train results')
    for n,(optic, sar_hr, sar_lr, res_label) in enumerate(trainloader):
        batch_tensor1 = None
        batch_tensor2 = None
        if n == 5: break
        if torch.cuda.is_available():
            optic = optic.cuda()
            sar_hr = sar_hr.cuda()
            sar_lr = sar_lr.cuda()
            res_label = res_label.cuda()

        with torch.no_grad():
            sar_sr,_ = srun_model(sar_lr, res_label)
            optic_gen_hr,_ = generator(F.interpolate(sar_hr, size=[256,256], mode='bicubic'))
            optic_gen_sr,_ = generator(F.interpolate(sar_sr, size=[256,256], mode='bicubic'))
            
        sar_lr_plot = F.interpolate(sar_lr, scale_factor=scale_factor, mode='nearest')
        if batch_tensor1 is None:
            batch_tensor1 = sar_lr_plot
        else:
            batch_tensor1 = torch.cat([batch_tensor1, sar_lr_plot], dim=0)
        batch_tensor1 = torch.cat([batch_tensor1, sar_sr], dim=0)
        batch_tensor1 = torch.cat([batch_tensor1, sar_hr], dim=0)
        
        if batch_tensor2 is None:
            batch_tensor2 = optic_gen_sr
        else:
            batch_tensor2 = torch.cat([batch_tensor2, optic_gen_sr], dim=0)
        batch_tensor2 = torch.cat([batch_tensor2, optic_gen_hr], dim=0)
        batch_tensor2 = torch.cat([batch_tensor2, optic], dim=0)

        grid_img = vutils.make_grid(batch_tensor1, nrow=6)
        plt.imshow(grid_img[0].squeeze().cpu().detach().numpy(), "gray")
        plt.savefig(path+'Generated_SAR_epoch'+str(epoch)+'_'+str(n)+'.png', dpi=350)
        
        # grid_img = vutils.make_grid(batch_tensor2, nrow=6)
        # plt.imshow(grid_img[0].squeeze().cpu().detach().numpy(), "gray")
        # plt.savefig(path+'Generated_Optic_epoch'+str(epoch)+'_'+str(n)+'.png', dpi=350)


def test_plotter(cfg, srun_model, generator, valloader, epoch, scale_factor, path):
    batch_tensor1 = None
    batch_tensor2 = None
    print('Plotting the SRUN Test results')
    number_cnt = 0
    for n,(optic,sar_hr,sar_lr, res_label) in enumerate(valloader):
        batch_tensor1 = None
        batch_tensor2 = None
        if number_cnt >= 5: break
        if random.random() > 0.95:
            if torch.cuda.is_available():
                optic = optic.cuda()
                sar_hr = sar_hr.cuda()
                sar_lr = sar_lr.cuda()
                res_label = res_label.cuda()
            with torch.no_grad():
                sar_sr,_ = srun_model(sar_lr, res_label)
                optic_gen_hr,_ = generator(F.interpolate(sar_hr, size=[256,256], mode='bicubic'))
                optic_gen_sr,_ = generator(F.interpolate(sar_sr, size=[256,256], mode='bicubic'))
                
            sar_lr_plot = F.interpolate(sar_lr, scale_factor=scale_factor, mode='nearest')
            if batch_tensor1 is None:
                batch_tensor1 = sar_lr_plot
            else:
                batch_tensor1 = torch.cat([batch_tensor1, sar_lr_plot], dim=0)
            batch_tensor1 = torch.cat([batch_tensor1, sar_sr], dim=0)
            batch_tensor1 = torch.cat([batch_tensor1, sar_hr], dim=0)
            
            if batch_tensor2 is None:
                batch_tensor2 = optic_gen_sr
            else:
                batch_tensor2 = torch.cat([batch_tensor2, optic_gen_sr], dim=0)
            batch_tensor2 = torch.cat([batch_tensor2, optic_gen_hr], dim=0)
            batch_tensor2 = torch.cat([batch_tensor2, optic], dim=0)
            number_cnt += 1

            grid_img = vutils.make_grid(batch_tensor1, nrow=6)
            plt.imshow(grid_img[0].squeeze().cpu().detach().numpy(), "gray")
            plt.savefig(path+'Generated_SAR_epoch'+str(epoch)+'_'+str(number_cnt)+'.png', dpi=350)
            
            # grid_img = vutils.make_grid(batch_tensor2, nrow=6)
            # plt.imshow(grid_img[0].squeeze().cpu().detach().numpy(), "gray")
            # plt.savefig(path+'Generated_Optic_epoch'+str(epoch)+'_'+str(number_cnt)+'.png', dpi=350)