import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, Whole_Slide_Bag_FP_Distort
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline

import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

import h5py
import openslide
import glob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    # dataset = Whole_Slide_Bag_FP_Distort(file_path=file_path, wsi=wsi, pretrained=pretrained,
    # 	custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    import pdb;
    pdb.set_trace()
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_downsample=custom_downsample, target_patch_size=256)
    x, y = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=8, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            import pdb; pdb.set_trace()
            batch = batch.to(device, non_blocking=True)
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()

if __name__ == '__main__':
    import openslide
    import tqdm
    import json
    sizes = []
    
    c = 0
    for wsi_path in tqdm.tqdm(glob.glob("/data3/CMCS/raw_data/Lymph_node_metasis_absent/*") + glob.glob("/data3/CMCS/raw_data/Lymph_node_metasis_absent/*")):
        try:
            sizes.append(openslide.open_slide(wsi_path).dimensions)
        except:
            c += 1

    print("Huge imgs:", c)