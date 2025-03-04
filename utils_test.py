import torch
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim
def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]

    ###### CH new code
    # ssim_list = []
    # for ind in range(len(dehaze_list)):
    #     # Get image dimensions
    #     # img_h, img_w, _ = dehaze_list_np[ind].shape
        
    #     # Define window size based on image dimensions
    #     # win_size = min(img_h, img_w, 7)  # 7 is the default window size; use the smaller dimension or 7
    #     # print(f"img_h: {img_h}, img_w: {img_w}, win_size: {win_size}")

    #     win_size = 3
        
    #     # Compute SSIM
    #     ssim_val = ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True, win_size=win_size)
    #     ssim_list.append(ssim_val)

    ########

    # ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True, win_size=3) for ind in range(len(dehaze_list))]  ## CH New code
    return ssim_list

def to_rmse(dehaze, gt):
    mse = torch.mean((dehaze - gt) ** 2, dim=[1, 2, 3])
    rmse = torch.sqrt(mse)
    rmse_list = [value.item() for value in rmse]
    return rmse_list



