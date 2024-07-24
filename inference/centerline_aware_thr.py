# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:38:03 2023

@author: q117mt
"""
import numpy as np
from skimage import measure
from utils.utils_imageclass import add_patch_to_image, threshold_img, get_patch
import matplotlib.pyplot as plt
from scipy import ndimage
from model.cldice_loss import soft_DiceLoss
from utils.utils_imageclass import getLargestCC
import torch
    
def add_patches_and_centerline_pathfinder_max(pred_patch_and_loc, shape, patch_size,
                                          thr = 0.5, thr_pathfinder = 0.1,
                                          dice = 0.9, ignore_isbelow = False,
                                          ignore_parallel_centerlines = False):
    
    diceloss = soft_DiceLoss()
    prediction = np.zeros(shape).astype(np.float64)
    count_seen = np.zeros(shape).astype(np.float64)
    centerline = np.zeros(shape).astype(np.float64)
    ones = np.ones_like(pred_patch_and_loc[0][0][0].squeeze().numpy())
    for k in range(len(pred_patch_and_loc)):
        for i in range(len(pred_patch_and_loc[k])):
            x_center = pred_patch_and_loc[k][i][1][-3]
            y_center = pred_patch_and_loc[k][i][1][-2]
            z_center = pred_patch_and_loc[k][i][1][-1]
            patch = pred_patch_and_loc[k][i][0].squeeze().numpy()
            prediction = add_patch_to_image(prediction, patch, x_center, y_center, z_center, patch_size)
            count_seen = add_patch_to_image(count_seen, ones, x_center, y_center, z_center, patch_size)
    
    if np.count_nonzero(prediction) == 0:
        return prediction, centerline, prediction
    prediction = prediction/count_seen.clip(min=1e-6)
    del count_seen
    prediction = threshold_img(prediction, thr)
    pred_lcc = getLargestCC(prediction)
    self_dice = 1-diceloss(torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0), 
                           torch.from_numpy(pred_lcc).unsqueeze(0).unsqueeze(0))
    
    prediction_max = np.zeros(shape).astype(np.float64)
    if  self_dice > dice:
        return prediction, centerline, prediction_max
    
    nonzero_zcomponent = np.nonzero(prediction)[2]
    nonzero_zcomponent_lcc = np.nonzero(pred_lcc)[2]
    z_min, z_max = np.min(nonzero_zcomponent), np.max(nonzero_zcomponent)
    z_min_lcc, z_max_lcc = np.min(nonzero_zcomponent_lcc), np.max(nonzero_zcomponent_lcc)
    
    if (z_min == z_min_lcc) and (z_max == z_max_lcc):
        return prediction, centerline, prediction_max
    
    else:
        for k in range(len(pred_patch_and_loc)):
            for i in range(len(pred_patch_and_loc[k])):
                x_center = pred_patch_and_loc[k][i][1][-3]
                y_center = pred_patch_and_loc[k][i][1][-2]
                z_center = pred_patch_and_loc[k][i][1][-1]
                patch = pred_patch_and_loc[k][i][0].squeeze().numpy()
                patch_cl = pred_patch_and_loc[k][i][-1]
                
                patch_seg = get_patch(prediction_max, x_center, y_center, z_center, patch_size=patch_size)
                prediction_max[x_center-patch_size:x_center+patch_size, 
                           y_center-patch_size:y_center+patch_size, 
                           z_center-patch_size:z_center+patch_size] = np.maximum(patch_seg, patch)
                
                
                if ignore_isbelow:
                    is_below = False
                else:
                    is_below = pred_patch_and_loc[k][i][3]
                
                is_above = pred_patch_and_loc[k][i][4]
                
                if (not is_below) and (not is_above):
                    if ignore_parallel_centerlines:
                        if (z_center-32 < z_min_lcc) or (z_center+32 > z_max_lcc):
                            centerline = add_patch_to_image(centerline, 
                                                            patch_cl, 
                                                            x_center, y_center, z_center, patch_size)
                    else:
                        centerline = add_patch_to_image(centerline, 
                                                        patch_cl, 
                                                        x_center, y_center, z_center, patch_size)
        
        

        prediction_max = threshold_img(prediction_max, thr_pathfinder)
        centerline[centerline > 0]  = 1
        tmp = centerline-centerline*prediction
        tmp = ndimage.binary_dilation(tmp, iterations = 9)
                
        z_component_nonzero_labels = np.nonzero(tmp)[-1]
        z_component_nonzero_labels = set(z_component_nonzero_labels)
        if len(z_component_nonzero_labels) == 0:
            for k in range(-8,8):
                if z_min != z_min_lcc:
                    prediction[:,:,z_min_lcc+k] = prediction_max[:,:,z_min_lcc+k]
                else:
                    prediction[:,:,z_max_lcc+k] = prediction_max[:,:,z_max_lcc+k]
        else:
            for k in set(z_component_nonzero_labels):
                prediction[:,:,k] = prediction_max[:,:,k]
        
            
        return prediction, centerline, prediction_max