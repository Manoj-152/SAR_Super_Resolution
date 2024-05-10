import torch

def minmax_scale(arr, new_min = 0., new_max = 1.):
    arr_min, arr_max = arr.min(), arr.max()
    epsilon = 1e-4
    if (arr_max == arr_min): 
        new_arr = ((arr - arr_min) / (arr_max - arr_min + epsilon)) * (new_max - new_min) + new_min
    else:
        new_arr = ((arr - arr_min) / (arr_max - arr_min)) * (new_max - new_min) + new_min
    return new_arr

def psnr(y_pred, y_true, minmax=True):
    if torch.isnan(y_pred.min()): print('y_pred before minmax')
    if torch.isnan(y_true.min()): print('y_true before minmax')
    
    if minmax == True:
        y_pred = minmax_scale(y_pred)
        y_true = minmax_scale(y_true)
        
    if torch.isnan(y_pred.min()): print('y_pred after minmax')
    if torch.isnan(y_true.min()): print('y_true after minmax')
    
    epsilon = 1e-6
    mse = torch.mean((y_pred - y_true)**2)
    if (torch.isnan(10 * torch.log10(1 / (mse + epsilon)))): print('NaN PSNR')
    return 10 * torch.log10(1 / (mse + epsilon))