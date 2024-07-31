# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:35:07 2023

@author: q117mt
"""

from inference.segmentation_function_inference import automatic_segmentation
import os
import numpy as np
import time
from utils.utils import get_logger
import torch
import nibabel

logger = get_logger('Pipeline')

path_to_data = r'./data/images'

shape = tuple((96,96,128))
patlist = os.listdir(path_to_data)

gauss_mus = [352,389,418]
gauss_sigma = [125, 147, 156]
in_channels = 4

folder_global = 'weights/weights_downsampled'
global_list = [os.path.join(folder_global, f) for f in os.listdir(folder_global)]
folder_local_1 = 'weights/weights_local/weights_label1'
weights_1 = [os.path.join(folder_local_1, f) for f in os.listdir(folder_local_1)]
folder_local_2 = 'weights/weights_local/weights_label2'
weights_2 = [os.path.join(folder_local_2, f) for f in os.listdir(folder_local_2)]
folder_local_3 = 'weights/weights_local/weights_label3'
weights_3 = [os.path.join(folder_local_3, f) for f in os.listdir(folder_local_3)]
folder_local_4 = 'weights/weights_local/weights_label4'
weights_4 = [os.path.join(folder_local_4, f) for f in os.listdir(folder_local_4)]

folder_local = folder_local_1.split('/')[1]

path_to_output = r'output'
if not os.path.isdir(path_to_output):
    os.mkdir(path_to_output)

path_to_nets = {'weights_global': global_list,
                'weights_1': weights_1,
                'weights_2': weights_2, 
                'weights_3': weights_3,
                'weights_4': weights_4
                }

thr_global = 0.25
thr_pathfinder_c = [0.25]
thr_pathfinder_v = [0.25]

thr_overall = 0.55
df = []

radius = 24
safety_break = 35
instancenorm = False

dic_pathfinder = dict()
flip_prediction = True
global_aware_pathfinder = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
self_dice = 0.975
thr_for_holes = 0.25
ignore_parallel_centerlines = True
add_max_style = True

use_direction_momentum = True
save_patches_aftertime = False
safe_result = True

patch_size = (32,32,32)
num_levels = 4
count = 0
times = []
thr_path = thr_pathfinder_c[0]
labels = [1,2,3,4]


input_dict = {'gauss_mus': gauss_mus, 'gauss_sigma': gauss_sigma, 'folder_global': folder_global, 
              'folder_local': folder_local, 'thr_global': thr_global,
                'thr_pathfinder_c': thr_path, 'thr_pathfinder_v': thr_path, 
                'thr_overall': thr_overall, 'in_channels': in_channels, 'flip_prediction': flip_prediction,
                'instancenorm': instancenorm, 'radius': radius, 'use_direction_momentum': use_direction_momentum, 
                'self_dice': self_dice, 'safety_break': safety_break,
                'thr_for_holes': thr_for_holes, 'ignore_parallel_centerlines': ignore_parallel_centerlines, 'add_max_style': add_max_style}
            
np.save(os.path.join(path_to_output, 'input_dict.npy'), input_dict)
time_ = []       

for pat in patlist:
    t1 = time.time()
    logger.info(f'Patient: {pat}')
    logger.info(f'Threshold pathfinder: {thr_path}')
    pred_patch_and_loc, affine, shape_big, all_seg = automatic_segmentation(pat, path_to_data, 
                                                                       gauss_mus, gauss_sigma, 
                                                                       shape, 
                                                                       path_to_nets,
                                                                       device = device,
                                                                       thr_global=thr_global,
                                                                       thr_pathfinder_c = thr_path,
                                                                       thr_pathfinder_v = thr_path,
                                                                       thr_overall=thr_overall,
                                                                       in_channels = in_channels,
                                                                       path_to_output = path_to_output, radius = radius,
                                                                       safety_break = safety_break,
                                                                       safeIt = False,
                                                                       patch_size = patch_size, num_levels = num_levels,
                                                                       instancenorm = instancenorm, 
                                                                       flip_prediction = flip_prediction,
                                                                       global_aware_pathfinder = global_aware_pathfinder, 
                                                                       self_dice = self_dice, thr_for_holes= thr_for_holes,
                                                                       ignore_parallel_centerlines = ignore_parallel_centerlines,
                                                                       add_max_style = add_max_style, 
                                                                       use_direction_momentum = use_direction_momentum, labels = labels,
                                                                       )

    
    if safe_result:
        if not os.path.isdir(os.path.join(path_to_output, 'segmentations')):
            os.mkdir(os.path.join(path_to_output, 'segmentations'))
        nifti_file = nibabel.Nifti1Image(all_seg.astype(np.uint8), affine)
        save_path = os.path.join(path_to_output,'segmentations', pat[:7] +'.nii.gz')
        nibabel.save(nifti_file, save_path)
        t2 = time.time()
        print(t2-t1)
        time_.append(t2-t1)
        print(np.mean(time_)/60)
        
        del nifti_file, save_path
                        
        if save_patches_aftertime:
            if not os.path.isdir(os.path.join(path_to_output, 'pred_patch_and_loc')):
                os.mkdir(os.path.join(path_to_output, 'pred_patch_and_loc'))
            save_path = os.path.join(path_to_output, 'pred_patch_and_loc', pat[:7])
            np.save(save_path, pred_patch_and_loc)


    del all_seg, pred_patch_and_loc