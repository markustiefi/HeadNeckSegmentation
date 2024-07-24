# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 09:50:46 2022

@author: A0067477
"""
import torch
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import fftpack


#Threshold the image
def threshold_img(image, threshold):
    '''
    Parameters
    ----------
    image : TYPE
        Input image.
    threshold : TYPE
        Threshold value.

    Returns
    -------
    image : TYPE
        Binary image.
        
    '''
    image[image < threshold] = 0
    image[image >= threshold] = 1
    return image

def mask(label):
    return np.ma.masked_where(label == 0, label)


def upsample(image, shape, mode='nearest'):
    '''

    Parameters
    ----------
    image : TYPE
        Downsampled image.
    shape : TYPE
        Output spatial shape.
    mode : TYPE, optional
        The upsampling algorithm: see torch.nn.Upsample documentation. 
        The default is 'nearest'.

    Returns
    -------
    TYPE
        Upsampled image.

    '''
    shape = tuple(shape)
    if len(shape)>3:
        shape = shape[-3::]
    up = torch.nn.Upsample(size = shape, mode=mode)
    return up(image)

def getLargestCC(segmentation, connectivity=1,  allow_break = False, factor = 1/2):
    '''

    Parameters
    ----------
    segmentation : TYPE
        Binary segmentation mask.

    Returns
    -------
    largestCC : TYPE
        Largest connected component of segmentation mask.

    '''
    if np.count_nonzero(segmentation) == 0:
        return segmentation
    
    #if np.bincount(labels.flat, weights = segmentation.flat).max() > th:
    #    allow_break = False
        
    if allow_break:
        labels = measure.label(segmentation)
        maximum = np.bincount(labels.flat, weights = segmentation.flat).max()
        max_th = maximum*factor
        largestCC = np.zeros_like(segmentation)
        ind = np.argpartition(np.bincount(labels.flat, weights=segmentation.flat), -2)[-3:]
        for index in ind:
            if np.bincount(labels.flat, weights=segmentation.flat)[index]>max_th:
                largestCC += labels == index
    else:
        labels, num = measure.label(segmentation, return_num = True, background = None, connectivity=connectivity)
        largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def get_patch(image, x_center, y_center, z_center, patch_size = 24,
              for_network = False):
    '''

    Parameters
    ----------
    image : TYPE
        Volume.
    z_center : int
    y_center : int
    x_center : int
    patch_size : TYPE, optional
        Size of patch to extract. The default is 24.
    for_network : TYPE, optional
        Add channel and batch dimension if needed (N, C, D, H, W). The default is False.

    Returns
    -------
    patch : TYPE
        Patch of image with size patch_size at given location.

    '''
    if isinstance(patch_size, int):
        patch = image[x_center-patch_size:x_center+patch_size, y_center-patch_size:y_center+patch_size, z_center-patch_size:z_center+patch_size]
    else:
        patch = image[x_center-patch_size[0]:x_center+patch_size[0], y_center-patch_size[1]:y_center+patch_size[1], z_center-patch_size[2]:z_center+patch_size[2]]
    if for_network:
        patch = patch.unsqueeze(0).unsqueeze(0)
    return patch

def add_patch_to_image(image, patch, z_center, y_center, x_center, patch_size):
    '''

    Parameters
    ----------
    image : TYPE
        Volume.
    patch : TYPE
        Patch of volume.
    z_center : int
    y_center : int
    x_center : int
    patch_size : TYPE
        Spatial size of patch.

    Returns
    -------
    image : TYPE
        Updated segmentation mask by adding current patch to volume.

    '''
    if isinstance(patch_size, int):
        image[z_center-patch_size:z_center+patch_size, y_center-patch_size:y_center+patch_size, x_center-patch_size:x_center+patch_size] += patch
    else:
        image[z_center-patch_size[0]:z_center+patch_size[0], y_center-patch_size[1]:y_center+patch_size[1], x_center-patch_size[2]:x_center+patch_size[2]] += patch
    return image

def get_global_index(local_point, global_point, patch_size):
    '''

    Parameters
    ----------
    local_point : TYPE
        Location of point within patch.
    global_point : TYPE
        Location of point within volume.
    patch_size : TYPE
        Spatial size of patch.

    Returns
    -------
    TYPE
        Location of point within volume with regard to patch size.

    '''
    if isinstance(patch_size, int):
        return local_point+global_point-[patch_size, patch_size, patch_size]
    else:
        return local_point+global_point-np.asarray(patch_size)

def create_one_hot(labels, num_classes, ignore_background = True):
    """
    Creates a one-hot encoded tensor from a tensor of labels.

    Parameters:
    labels (torch.Tensor): A tensor of integer labels of shape (N, *), where N is the batch size
                           and * represents any number of additional dimensions.
    num_classes (int): The number of classes.

    Returns:
    torch.Tensor: A one-hot encoded tensor of shape (N, *, num_classes).
    """
    # Ensure the labels are of integer type
    labels = labels.long()

    # Get the shape of the labels tensor
    label_shape = labels.shape

    # Create a one-hot encoded tensor of zeros with an additional dimension for classes
    one_hot = torch.zeros(*label_shape, num_classes, dtype=torch.float32, device=labels.device)

    # Scatter 1s to the appropriate locations
    one_hot.scatter_(-1, labels.unsqueeze(-1), 1)
    if ignore_background:
        return one_hot[1::]
    else:
        return one_hot

    
def get_start_point_list_fast(prediction, shape, global_thr = 0.5):
    
    prediction = np.uint(getLargestCC(threshold_img(np.asarray(prediction), global_thr), allow_break=True))*np.array(prediction)
    z_min, z_max = np.where(prediction!=0)[-1].min(), np.where(prediction)[-1].max()
    
    if (z_max-z_min)>32:
        prediction[:,:,0:(z_min+8)] = 0
        prediction[:,:,(z_max-8):] = 0
    
    x_list, y_list, z_list = np.unravel_index(np.argsort(prediction.ravel()), prediction.shape)
    listed_indices = list(zip(x_list, y_list, z_list))
    n = 2
    while (np.abs(prediction[listed_indices[-1]]-prediction[listed_indices[-n]]) < 0.3) & (prediction[listed_indices[-n]] >= 0.75):
        n += 1
    n -= 1 
    
    listed_indices_red = listed_indices[-n:-1]
    argmax = np.array(listed_indices_red[-1])
    list_dist = []
    for i in range(len(listed_indices_red)):
        arg_tmp = np.array(listed_indices_red[i])
        dist = np.linalg.norm(argmax-arg_tmp)**2
        list_dist.append(dist)
        
    list_dist_sorted = np.sort(list_dist)
    ind = np.where(list_dist == list_dist_sorted[int(len(list_dist)*90/100)])[0][0]
    #ind = np.argmax(list_dist)

    location_1 = ((np.asarray(argmax)/prediction.shape)*shape[1:]).astype(np.int16)
    location_2 = ((np.asarray(listed_indices_red[ind])/prediction.shape)*shape[1:]).astype(np.int16)

    start_point_list = [location_1, location_2]
    return start_point_list, [], []