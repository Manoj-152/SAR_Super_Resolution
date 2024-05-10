from Analysis.plotters import train_plotter, test_plotter
from Analysis.SSIM import ssim
from Analysis.Performance_Measures import psnr

from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(10)

def save_weights(path,optimizer,scheduler,model,epoch):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }, path)
    

# Function used for training the model
def train(cfg, trainloader, valloader, trainloader_plot, valloader_plot, srun_model, generator, optimizer, scheduler, start_epoch):
    os.makedirs(cfg['RESULT_DIRS']['WEIGHTS'], exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'], exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/SuperResolution_Train_Results', exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/SuperResolution_Test_Results', exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['LOSSES'], exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['ACCURACIES'], exist_ok=True)
    
    saving_after_epochs = cfg['TRAIN']['SAVING_AFTER_EPOCHS']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg['TRAIN']['NUM_EPOCHS']
    if os.path.isfile(cfg['RESULT_DIRS']['WEIGHTS'] + "/best_ckpt.pth"):
        best_ckpt = torch.load(cfg['RESULT_DIRS']['WEIGHTS'] + "/best_ckpt.pth")
        lowest_validation_loss = best_ckpt['Validation Loss']
        print('Lowest Validation Loss till now: ', lowest_validation_loss)
    else:
        lowest_validation_loss = 10000.
        print('No best checkpoint received')
        
    srun_model = srun_model.to(device)
    generator = generator.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # Starting training
        running_loss = 0.
        running_content_loss = 0.
        running_eval_loss = 0.
        running_psnr = 0.
        running_ssim = 0.
        
        for n,(optic,sar_hr,sar_lr,res_label) in enumerate(tqdm(trainloader)):
            if n == len(trainloader) - 1: break
            optic = optic.to(device)
            sar_hr = sar_hr.to(device)
            sar_lr = sar_lr.to(device)
            res_label = res_label.to(device)
    
            optimizer.zero_grad()
            batch_size = optic.size(0)
    
            if cfg['RESOLUTION_METHOD'] == 'Two Step':
                sar_sr,_ = srun_model(sar_lr, res_label)
            else:
                sar_sr,_ = srun_model(sar_lr)
            # Content Loss -> L1 Loss between sar_hr and sar_sr
            content_loss = F.l1_loss(sar_sr, sar_hr)
    
            # Evaluation loss -> L1 loss between l_hr and l_sr
            with torch.no_grad():
                optical_gen_hr,_ = generator(F.interpolate(sar_hr, size=[256,256], mode='bicubic'))       # sar_hr is the ground truth, there is no need to compute gradients for it as ground truth is independent of the weights
            optical_gen_sr,_ = generator(F.interpolate(sar_sr, size=[256,256], mode='bicubic'))
            # l_hr is the L1 loss between optical gt and optical generated from sar_hr
            # l_sr is the L1 loss between optical gt and optical generatd from sar_sr
            l_hr = F.l1_loss(optical_gen_hr, optic)
            l_sr = F.l1_loss(optical_gen_sr, optic)
            evaluation_loss = F.l1_loss(l_hr, l_sr)
    
            lambda_var = cfg['TRAIN']['EVAL_LOSS_WEIGHT']
            total_loss = content_loss + lambda_var*evaluation_loss
            
            total_loss.backward()
            optimizer.step()
            running_content_loss += content_loss.item()
            running_eval_loss += evaluation_loss.item()
            running_loss += total_loss.item()
            
            ssim_score = ssim(sar_sr, sar_hr, val_range=2, minmax=True)
            psnr_score = psnr(sar_sr, sar_hr, minmax=True)
            running_ssim += ssim_score.item()
            running_psnr += psnr_score.item()
            
        with open(cfg['RESULT_DIRS']['LOSSES'] + '/content_train_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_content_loss / (len(trainloader) - 1)))
        with open(cfg['RESULT_DIRS']['LOSSES'] + '/eval_train_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_eval_loss / (len(trainloader) - 1)))
        with open(cfg['RESULT_DIRS']['LOSSES'] + '/overall_train_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_loss / (len(trainloader) - 1)))
        with open(cfg['RESULT_DIRS']['ACCURACIES'] + '/ssim_score_train.txt', 'a') as f:
            f.write("%s\n" % str(running_ssim / (len(trainloader) - 1)))
        with open(cfg['RESULT_DIRS']['ACCURACIES'] + '/psnr_score_train.txt', 'a') as f:
            f.write("%s\n" % str(running_psnr / (len(trainloader) - 1)))
    
        print('Epoch: ',epoch,' Content_Train_loss: ',round(running_content_loss / (len(trainloader) - 1), 4), ' Eval_Train_loss: ', round(running_eval_loss/ (len(trainloader)-1),4), ' Overall_Train_loss: ', round(running_loss/(len(trainloader)-1),4), ' SSIM score: ',round(running_ssim / (len(trainloader) - 1), 4), \
                                                    ' PSNR score: ',round(running_psnr / (len(trainloader) - 1), 4))
    
        if epoch%saving_after_epochs == 0 or epoch == num_epochs - 1:
            save_weights(cfg['RESULT_DIRS']['WEIGHTS'] + "/Epoch_"+str(epoch)+".pth",optimizer,scheduler,srun_model,epoch)
            
        running_loss = 0.
        running_content_loss = 0.
        running_eval_loss = 0.
        running_psnr = 0.
        running_ssim = 0.
    
        for n,(optic,sar_hr,sar_lr,res_label) in enumerate(tqdm(valloader)):
            if n == len(valloader) - 1: break
            optic = optic.to(device)
            sar_hr = sar_hr.to(device)
            sar_lr = sar_lr.to(device)
            res_label = res_label.to(device)
            
            batch_size = optic.size(0)
            with torch.no_grad():
                sar_sr,_ = srun_model(sar_lr, res_label)
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
            
        with open(cfg['RESULT_DIRS']['LOSSES'] + '/content_test_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_content_loss / (len(valloader) - 1)))
        with open(cfg['RESULT_DIRS']['LOSSES'] + '/eval_test_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_eval_loss / (len(valloader) - 1)))
        with open(cfg['RESULT_DIRS']['LOSSES'] + '/overall_test_loss.txt', 'a') as f:
            f.write("%s\n" % str(running_loss / (len(valloader) - 1)))
        with open(cfg['RESULT_DIRS']['ACCURACIES'] + '/ssim_score_test.txt', 'a') as f:
            f.write("%s\n" % str(running_ssim / (len(valloader) - 1)))
        with open(cfg['RESULT_DIRS']['ACCURACIES'] + '/psnr_score_test.txt', 'a') as f:
            f.write("%s\n" % str(running_psnr / (len(valloader) - 1)))
    
        print('Epoch: ',epoch,' Content_Test_loss: ',round(running_content_loss / (len(valloader) - 1), 4), ' Eval_Test_loss: ', round(running_eval_loss/ (len(valloader)-1),4), ' Overall_Test_loss: ', round(running_loss/(len(valloader)-1),4), ' SSIM score: ',round(running_ssim / (len(valloader) - 1), 4), \
                                                    ' PSNR score: ',round(running_psnr / (len(valloader) - 1), 4))
                                                    
        if (running_loss / (len(valloader) - 1)) < lowest_validation_loss:
            print("Saving best checkpoint, Epoch ", epoch)
            lowest_validation_loss = running_loss / (len(valloader) - 1)
            torch.save({
                      "model": srun_model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "epoch": epoch,
                      "Validation Loss": running_loss / (len(valloader) - 1)}, cfg['RESULT_DIRS']['WEIGHTS'] + "/best_ckpt.pth")
                      
        scheduler.step(running_loss / (len(valloader) - 1))
        
        if cfg['RESOLUTION_METHOD'] == 'Two Step':
            train_plotter(cfg, srun_model, generator, trainloader_plot, epoch, 4, cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/SuperResolution_Train_Results/')
            test_plotter(cfg, srun_model, generator, valloader_plot, epoch, 4, cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/SuperResolution_Test_Results/')
        else:
            train_plotter(cfg, srun_model, generator, trainloader_plot, epoch, cfg['SCALE_RATIO'], cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/SuperResolution_Train_Results/')
            test_plotter(cfg, srun_model, generator, valloader_plot, epoch, cfg['SCALE_RATIO'], cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/SuperResolution_Test_Results/')
        
        # if epoch%20 == 0: plot_features(srun_model, trainloader, epoch, cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/')