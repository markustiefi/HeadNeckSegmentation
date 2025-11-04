# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:30:54 2023

@author: q117mt
"""

import os
import torch
import nibabel
import numpy as np
import torchio as tio
from inference.Downsample_class import Downsample
from inference.Tracking_class import Image_predict

from utils.utils_imageclass import getLargestCC
from inference.centerline_aware_thr import add_patches_and_centerline_pathfinder_max
from utils.utils import get_logger
import asyncio
import nest_asyncio
nest_asyncio.apply()

logger = get_logger('Segmentation function')

def automatic_segmentation(pat, path_to_data, 
                           gauss_mus, gauss_sigma,
                           shape,
                           path_to_nets,
                           device,
                           thr_global, 
                           thr_pathfinder_c,
                           thr_pathfinder_v,
                           thr_overall, 
                           in_channels,
                           path_to_output='predictions',
                           safeIt = False, 
                           patch_size = 32, num_levels = 4, radius = 12, 
                           safety_break = 40, 
                           instancenorm = True, flip_prediction = True, global_aware_pathfinder =True,
                           self_dice = 0.9, thr_for_holes = 0.1, ignore_parallel_centerlines = True,
                           add_max_style = False, use_direction_momentum = False,
                           in_channels_local = 1, out_channels_local = 1, labels = [1,2,3,4], allow_break = True,
                           ):

    #time1 = time()
    
    path_to_global = path_to_nets['weights_global']

    all_segmentations = None


    subject = Downsample(pat, path_to_data,
                                      gauss_mus, gauss_sigma, 
                                       device,
                                       shape).pred_downsampled(path_to_global, in_channels = in_channels, 
                                                               out_channels = 5, 
                                                               final_sigmoid = False
                                                               )
        
    if pat.endswith('.nii.gz'):
        pat_id = pat[:-7]
    

    
    
    
    pred_patch_and_loc = Image_predict(pat, path_to_data, 
                                       gauss_mus, gauss_sigma, 
                                       shape, device,
                                       path_to_global, ' ', 
                                       ' ', path_to_nets, 
                                       thr_global, safety_break, instancenorm, 
                                       flip_prediction,
                                       global_aware_pathfinder, 
                                       use_direction_momentum,
                                       in_channels_local=in_channels_local, 
                                       out_channels_local=out_channels_local,
                                       labels = labels).segnfollow_v2(subject = subject, 
                                                                              patch_size = patch_size,
                                                                              radius = radius,
                                                                              thr_pathfinder_c=thr_pathfinder_c, 
                                                                              thr_pathfinder_v =thr_pathfinder_v,
                                                                              thr_overall = thr_overall)
    
    #time2 = time()
    #t = (time2-time1)/60
    #logger.info(f'Time: {t} min')
    shape = subject.image.shape[1::]
    subject.remove_image('prediction_downsampled')
    subject.remove_image('prediction_upsampled')

    loop = asyncio.get_event_loop()
    looper = asyncio.gather(*[add_patches_parallel(pred_patch_and_loc, key, thr_pathfinder_c, thr_pathfinder_v, thr_overall, add_max_style,
                             thr_for_holes, shape, self_dice, ignore_parallel_centerlines, subject, allow_break) for key in pred_patch_and_loc.keys()])
    all_seg = loop.run_until_complete(looper)
    
    all_segmentations = np.zeros(shape)
    for i, [t, key] in enumerate(all_seg):
        t[all_segmentations != 0] = 0
        all_segmentations += int(key[-1])*t.astype('uint8')
        
    if subject.was_resized:
        resize = tio.Resize(subject.shape_before_resizing, image_interpolation = 'nearest')
        all_segmentations = resize(all_segmentations)
        print(all_segmentations.shape)
    
        
    if safeIt:
        if not os.path.isdir(os.path.join(path_to_output, 'segmentations')):
            os.mkdir(os.path.join(path_to_output, 'segmentations'))
        all_segmentations = torch.rot90(torch.tensor(all_segmentations), k = -2, dims = [0,1]).numpy()
        affine = subject.image.affine
        nifti_file = nibabel.Nifti1Image(all_segmentations.astype(np.uint8), affine)
        save_path = os.path.join(path_to_output,'segmentations', pat_id)
        nibabel.save(nifti_file, save_path)
        del nifti_file, save_path
    
    shape = subject.image.shape
    affine = subject.image.affine
    return pred_patch_and_loc, affine, shape, all_segmentations

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

@background
def add_patches_parallel(pred_patch_and_loc, key, thr_pathfinder_c, thr_pathfinder_v, thr_overall, add_max_style,
                         thr_for_holes, shape, self_dice, ignore_parallel_centerlines, subject, allow_break):
    i = int(key[-1])-1
    if i<2:
        thr_pathfinder = thr_pathfinder_c
    else:
        thr_pathfinder = thr_pathfinder_v


    if i<2:
        ignore_isbelow = False
    else:
        ignore_isbelow = True
        
    

    pred_mean, _, prediction_pathfinder = add_patches_and_centerline_pathfinder_max(pred_patch_and_loc[key], shape, 32, thr = thr_overall, 
                                                                                    thr_pathfinder = thr_for_holes, dice = self_dice,
                                                                                    ignore_isbelow=ignore_isbelow,
                                                                                    ignore_parallel_centerlines = ignore_parallel_centerlines)

    if np.count_nonzero(pred_mean) == 0:
        pred_mean_lcc = pred_mean
    else:
        pred_mean_lcc = getLargestCC(pred_mean, allow_break = allow_break)
        
    return [pred_mean_lcc, key]

