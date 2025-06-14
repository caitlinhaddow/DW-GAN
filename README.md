# CH Notes
File structure for training
- test_data directory containing clean and hazy directories, and test.txt
- training_data directory containing clean and hazy directories, and train.txt
- added train_dataset.py based on file from ITBDehaze
- check_points (where models saved to during training)
- added .py file for generating the txts

Original authors trained using
- 2 GPUs (nvidia 1080Ti gpu, assumed 11GB each)
- 8000 epochs, batch size 16
- Random data augmentation (90, 180, 270 rotation and horizontal flip)
- Achieved final performance with a mix of NH-Haze (gamma corrected) and NHHaze2

# DW-GAN: A discrete wavelet transform GAN for NonHomogenous Image Dehazing - NTIRE 2021

This is the official PyTorch implementation of DW-GAN.  
1st place solution of NTIRE 2021 NonHomogeneous Dehazing Challenge (CVPR Workshop 2021).

Email contact: fig072@usask.ca

See more details in  [[report]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Ancuti_NTIRE_2021_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2021_paper.pdf) , [[paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Fu_DW-GAN_A_Discrete_Wavelet_Transform_GAN_for_NonHomogeneous_Dehazing_CVPRW_2021_paper.pdf), [[certificates]](https://data.vision.ee.ethz.ch/cvl/ntire21/NTIRE2021awards_certificates.pdf)

## Environment:

- Ubuntu: 18.04

- CUDA Version: 11.0 
- Python 3.8

## Dependencies:

- torch==1.6.0
- torchvision==0.7.0
- NVIDIA GPU and CUDA

## Pretrained Weights & Dataset

1. Download [ImageNet pretrained weights](https://drive.google.com/file/d/1aZQyF16pziCxKlo7BvHHkrMwb8-RurO_/view?usp=drive_link) and [Dehaze weights](https://drive.google.com/file/d/1h_HaRWwvx-ViCCKJK9oODlwS_lWNxt4B/view?usp=sharing) and place into the folder ```./weights```.  
2. Download the [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) and [NH-HAZE2](https://competitions.codalab.org/competitions/28032#participate) dataset.

## Test

For inference, run following commands. Please check the test hazy image path (test.py line 12) and the output path (test.py line 13) .

```
python test.py
```

## Qualitative Results

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge validation images:  

<div style="text-align: center">
<img alt="" src="/Image/validation.png" style="display: inline-block;" />
</div>

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:

<div style="text-align: center">
<img alt="" src="/Image/test.png" style="display: inline-block;" />
</div>


## Acknowledgement

We thank the authors of [Res2Net](https://mmcheng.net/res2net/), [MWCNN](https://github.com/lpj0/MWCNN.git), and [KTDN](https://github.com/GlassyWu/KTDN). Part of our code is built upon their modules.

## Citation

If our work helps your research, please consider to cite our paper:

```
@InProceedings{Fu_2021_CVPR,
    author    = {Fu, Minghan and Liu, Huan and Yu, Yankun and Chen, Jun and Wang, Keyan},
    title     = {DW-GAN: A Discrete Wavelet Transform GAN for NonHomogeneous Dehazing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {203-212}
}
```



