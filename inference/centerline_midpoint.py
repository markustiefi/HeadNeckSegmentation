# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:10:06 2022

@author: A0067477
"""
import numpy as np
from skimage.morphology import skeletonize_3d

def get_centerline(patch_pred):
    return skeletonize_3d(patch_pred)

def get_new_midpoints_fast(binary, projected_mid = [], radius =8, patch_size = 24):

    centerline = get_centerline(binary)
    nonzero_centerline = np.argwhere(centerline != 0)
    if isinstance(patch_size, int):
       patch_center = np.asarray([patch_size]*3)
    else:
        patch_center = np.asarray(patch_size)
    
    if len(nonzero_centerline) == 0:
        print('Empty prediciton')
        if isinstance(patch_size, int):
            return patch_center, patch_center, [], [], []
        else:
            return patch_center, patch_center, [], [], []
    
    norm_projectedmid = np.linalg.norm(nonzero_centerline-patch_center, axis = 1)
    ind_min = np.argmin(norm_projectedmid)
    projected_mid = nonzero_centerline[ind_min]
    
    norm_ball = np.linalg.norm(nonzero_centerline-projected_mid, axis = 1)
    dist_list = []
    
    acceptence_rate = 2
    norm_max = 0
    for i, x in enumerate(norm_ball):
        if (x >= radius) and (x<=radius+acceptence_rate):
            if len(dist_list)==0:
                dist_list.append(nonzero_centerline[i])
            else:
                for element in dist_list:
                    norm_tmp = np.linalg.norm(element-nonzero_centerline[i])
                    if norm_tmp > np.min([acceptence_rate+2,radius/2]):
                        if norm_tmp > norm_max:
                            norm_max = norm_tmp
                            dist_list.insert(0, element)
                            dist_list.insert(1, nonzero_centerline[i])
    

    if len(dist_list) >= 2:
        xc1 = dist_list[0]
        xc2 = dist_list[1]
    elif len(dist_list) == 1:
        xc1 = dist_list[0]
        xc2 = dist_list[0]
    else:
        xc1 = patch_center
        xc2 = patch_center
    
    return xc1, xc2, np.zeros_like(centerline), centerline, projected_mid

