# External imports
import torch
import argparse
import torch.nn as nn
from torchvision.utils import save_image as imwrite
import os
import time
# import re
from tqdm import tqdm ########## ---- CH code ---- ##########
from collections import OrderedDict ########## ---- CH code ---- ##########
import string

# Internal imports
from torch.utils.data import DataLoader
from test_dataset import dehaze_test_dataset
from model import fusion_net

## ch add
def normalize_state_dict(state_dict):
    """" Remove .module prefixes """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict

parser = argparse.ArgumentParser(description='Dehaze')
parser.add_argument('--test_dir', type=str, default='./Please load your hazy image path/')
parser.add_argument('--output_dir', type=str, default='./output_result/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--datasets', nargs='+', default=['Test'])  ## ch add
parser.add_argument('--weights', nargs='+', default=['dehaze.pkl'], help='List of weight file names as strings') ## ch add
args = parser.parse_args()

# test_dir = args.test_dir
test_batch_size = args.test_batch_size
output_dir = args.output_dir

for dataset in args.datasets:
    print(f"-- Testing on dataset {dataset}: --")
    test_data_dir = os.path.join(args.test_dir, dataset)  ## ch add  
    list_weight_files = args.weights ## ch add

    test_dataset = dehaze_test_dataset(test_data_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=2)
    tested_on = dataset ## ch add

    for weight_file in list_weight_files:
        weight_name = os.path.splitext(os.path.basename(weight_file))[0]
        final_output_dir = os.path.join(output_dir, f"{tested_on}_{weight_name}")  # save each set of images in a new directory with the weight name

        weight_file = os.path.join("./weights", weight_file)  ## weights must be saved in weights directory
        ########## ---- End of CH code ---- ##########

        ########## ---- Start of CH code: To use given pkl file with parallel GPUs ---- ##########
        multiple_gpus = True  ## SET VARIABLE
        if multiple_gpus:
            # --- Gpu device --- #
            device_ids = [Id for Id in range(torch.cuda.device_count())]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # --- Define the network --- #
            net = fusion_net().to(device)

            # Load checkpoint first, without wrapping in DataParallel
            checkpoint = torch.load(weight_file, map_location=device)
            print(f"Using weights from: {weight_file}")

            if "model_state_dict" in checkpoint:
                print(f"found model state dict")
                state_dict = checkpoint["model_state_dict"]  # Extract the actual state dict
            elif "state_dict" in checkpoint:
                print(f"found state dict")
                state_dict = checkpoint["state_dict"] 
            else:
                state_dict = checkpoint  # Direct state_dict case
                print(f"no model state dict")

            # Remove "module." prefix if it exists
            new_state_dict = normalize_state_dict(state_dict)

            # Load state dict into model
            net.load_state_dict(new_state_dict)
            ########## ---- End of CH code ---- ##########

            # Now wrap in DataParallel
            net = nn.DataParallel(net, device_ids=device_ids)

        else:
            ########## ---- Start of CH code: To get around parallel GPU requirement ---- ##########
            print("Using single GPU only")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = fusion_net().to(device)

            # Load checkpoint
            ckpt_path = weight_file
            
            checkpoint = torch.load(ckpt_path, map_location=device)  # Ensure checkpoint is loaded on correct device

            if "model_state_dict" in checkpoint:
                print(f"found model state dict")
                state_dict = checkpoint["model_state_dict"]  # Extract the actual state dict
            elif "state_dict" in checkpoint:
                print(f"found state dict")
                state_dict = checkpoint["state_dict"] 
            else:
                state_dict = checkpoint  # Direct state_dict case
                print(f"no model state dict")

            # Remove "module." prefix if it exists
            new_state_dict = normalize_state_dict(state_dict)

            # Load state dict into model
            net.load_state_dict(new_state_dict)
            ########## ---- End of CH code ---- ##########

        # --- Test --- #
        with torch.no_grad():
            net.eval()
            start_time = time.time()
            for batch_idx, (hazy_up,hazy_down,name) in enumerate(tqdm(test_loader)):
                hazy_up = hazy_up.to(device)
                hazy_down = hazy_down.to(device)
                frame_out_up = net(hazy_up)
                frame_out_down = net(hazy_down)
                frame_out = (torch.cat([frame_out_up[:, :, 0:600, :].permute(0, 2, 3, 1), frame_out_down[:, :, 552:, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)
                if not os.path.exists(final_output_dir + '/'):
                    os.makedirs(final_output_dir + '/')

                ########## ---- Start of CH code: Output meaningful filenames ---- ##########
                cleaned_filename = str(name).strip(string.punctuation)
                # print(str(name))
                imwrite(frame_out, os.path.join(final_output_dir, cleaned_filename), range=(0, 1))
                ########## ---- End of CH code ---- ##########

        test_time = time.time() - start_time
        print(test_time)












