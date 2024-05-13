from Analysis.SSIM import ssim
from Analysis.Performance_Measures import psnr

import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
torch.manual_seed(10)
random.seed(42)    

def plot_validation(sar_lr, sar_sr, sar_hr, scale_factor, plot_cnt, path):
    sar_lr_plot = F.interpolate(sar_lr, scale_factor=scale_factor, mode='nearest')
    batch_tensor1 = sar_lr_plot
    batch_tensor1 = torch.cat([batch_tensor1, sar_sr], dim=0)
    batch_tensor1 = torch.cat([batch_tensor1, sar_hr], dim=0)
    grid_img = vutils.make_grid(batch_tensor1, nrow=3)
    plt.imshow(grid_img[0].squeeze().cpu().detach().numpy(), "gray")
    plt.savefig(path+'Generated_SAR_epoch_'+str(plot_cnt)+'.png', dpi=350)


# Function used for validating the model (singlepass)
def validate_singlepass(cfg, valloader, srun_model, generator):
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'], exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/Validation_Results', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_ckpt = torch.load(cfg['SRUN_VALIDATION_CKPT'])
    lowest_validation_loss = best_ckpt['Validation Loss']
    print(f'Lowest Validation Loss till now: {lowest_validation_loss}. Obtained at epoch {best_ckpt["epoch"]}')
        
    srun_model = srun_model.to(device)
    generator = generator.to(device)
    
    running_loss = 0.
    running_content_loss = 0.
    running_eval_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    plot_cnt = 0

    for n,(optic,sar_hr,sar_lr,res_label) in enumerate(tqdm(valloader)):
        if n == len(valloader) - 1: break
        optic = optic.to(device)
        sar_hr = sar_hr.to(device)
        sar_lr = sar_lr.to(device)
        res_label = res_label.to(device)
        
        batch_size = optic.size(0)
        with torch.no_grad():
            if cfg['RESOLUTION_METHOD'] == 'Two Step':
                sar_sr,_ = srun_model(sar_lr, res_label)
            else:
                sar_sr,_ = srun_model(sar_lr)
        content_loss = F.l1_loss(sar_sr, sar_hr)
        
        with torch.no_grad():
            optical_gen_hr,_ = generator(F.interpolate(sar_hr, size=[256,256], mode='bicubic'))
            optical_gen_sr,_ = generator(F.interpolate(sar_sr, size=[256,256], mode='bicubic'))
        l_hr = F.l1_loss(optical_gen_hr, optic)
        l_sr = F.l1_loss(optical_gen_sr, optic)
        evaluation_loss = F.l1_loss(l_hr, l_sr)
        
        lambda_var = cfg['TRAIN']['EVAL_LOSS_WEIGHT']
        total_loss = content_loss + lambda_var*evaluation_loss
        
        running_content_loss += content_loss.item()
        running_eval_loss += evaluation_loss.item()
        running_loss += total_loss.item()
        
        ssim_score = ssim(sar_sr, sar_hr, val_range=2, minmax=True)
        psnr_score = psnr(sar_sr, sar_hr, minmax=True)
        running_ssim += ssim_score.item()
        running_psnr += psnr_score.item()

        if random.random() > 0.99 and plot_cnt < 20:
            plot_cnt += 1
            plot_validation(sar_lr, sar_sr, sar_hr, cfg['SCALE_RATIO'], plot_cnt, cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/Validation_Results/')

    print('Validation: Content_Test_loss: ',round(running_content_loss / (len(valloader) - 1), 4), ' Eval_Test_loss: ', round(running_eval_loss/ (len(valloader)-1),4), ' Overall_Test_loss: ', round(running_loss/(len(valloader)-1),4), ' SSIM score: ',round(running_ssim / (len(valloader) - 1), 4), \
                                                ' PSNR score: ',round(running_psnr / (len(valloader) - 1), 4))
                                                

# Function used for validating the model (2step)
def validate_2step(cfg, valloader, srun_model, generator):
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'], exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/Validation_Results', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_ckpt = torch.load(cfg['SRUN_VALIDATION_CKPT'])
    lowest_validation_loss = best_ckpt['Validation Loss']
    print(f'Lowest Validation Loss till now: {lowest_validation_loss}. Obtained at epoch {best_ckpt["epoch"]}')

    srun_model = srun_model.to(device)
    generator = generator.to(device)
    
    running_loss = 0.
    running_content_loss = 0.
    running_eval_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    plot_cnt = 0

    for n,(optic,sar_hr,sar_lr) in enumerate(tqdm(valloader)):
        if n == len(valloader) - 1: break
        optic = optic.to(device)
        sar_hr = sar_hr.to(device)
        sar_lr = sar_lr.to(device)
        batch_size = optic.size(0)

        res_label1 = torch.ones(batch_size)
        res_label1 = res_label1.unsqueeze(dim=1).to(device) 
        res_label2 = torch.ones(batch_size)*4
        res_label2 = res_label2.unsqueeze(dim=1).to(device)
        with torch.no_grad(): 
            sar_sr_mid, _ = srun_model(sar_lr, res_label1)
            sar_sr, _ = srun_model(sar_sr_mid, res_label2)

        content_loss = F.l1_loss(sar_sr, sar_hr)
        
        with torch.no_grad():
            optical_gen_hr,_ = generator(F.interpolate(sar_hr, size=[256,256], mode='bicubic'))
            optical_gen_sr,_ = generator(F.interpolate(sar_sr, size=[256,256], mode='bicubic'))
        l_hr = F.l1_loss(optical_gen_hr, optic)
        l_sr = F.l1_loss(optical_gen_sr, optic)
        evaluation_loss = F.l1_loss(l_hr, l_sr)
        
        lambda_var = cfg['TRAIN']['EVAL_LOSS_WEIGHT']
        total_loss = content_loss + lambda_var*evaluation_loss
        
        running_content_loss += content_loss.item()
        running_eval_loss += evaluation_loss.item()
        running_loss += total_loss.item()
        
        ssim_score = ssim(sar_sr, sar_hr, val_range=2, minmax=True)
        psnr_score = psnr(sar_sr, sar_hr, minmax=True)
        running_ssim += ssim_score.item()
        running_psnr += psnr_score.item()

        if random.random() > 0.99 and plot_cnt < 20:
            plot_cnt += 1
            plot_validation(sar_lr, sar_sr, sar_hr, 16, plot_cnt, cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/Validation_Results/')

    print('Validation: Content_Test_loss: ',round(running_content_loss / (len(valloader) - 1), 4), ' Eval_Test_loss: ', round(running_eval_loss/ (len(valloader)-1),4), ' Overall_Test_loss: ', round(running_loss/(len(valloader)-1),4), ' SSIM score: ',round(running_ssim / (len(valloader) - 1), 4), \
                                                ' PSNR score: ',round(running_psnr / (len(valloader) - 1), 4))