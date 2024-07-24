# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:07:15 2023

@author: q117mt
"""

import os
import torchio as tio
from torch.utils.data import DataLoader

from utils.utils import get_logger

logger = get_logger('Queue')

class window_image():
    def __init__(self, window_center = 320, window_width = 1000):
        self.window_center = window_center
        self.window_width = window_width
    
    def window(self, image):
        img_min = self.window_center-self.window_width//2
        img_max = self.window_center+self.window_width//2
        window_image = image.clone()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image
    
def labelsampler(patch_size = 64, label = 1, prob_positive = 8):
    prob_negative = 10-prob_positive
    sampler = tio.data.LabelSampler(patch_size = patch_size, 
                                    label_probabilities = {0:prob_negative, label:prob_positive})
    return sampler


def Queue(list_raw, list_label, batch_size = 4, patch_size = 48, label = 1, queue_length = 50, 
          samples_per_volume = 100, train = True, prob_positive = 8,
          window_center = 320, window_width = 1000, flipit = False):
    
    # Create a torchio subject dataset.
    subject_list = []
    for i in range(len(list_raw)):
        subject = tio.Subject(image = tio.ScalarImage(list_raw[i]),
                              label = tio.LabelMap(list_label[i]),
                              patient = os.path.basename(list_raw[i]).split('.')[0])
        subject_list.append(subject)
        
    
    # Preprocessing and further transforms for training.
    transforms_preprocessing = [
        tio.ToCanonical(),
        tio.Lambda(window_image(window_center = window_center, window_width = window_width).window),
        tio.RescaleIntensity(out_min_max = (0,1)),
        tio.CopyAffine('image'),
        ]
    
    transform_preprocess = tio.Compose(transforms_preprocessing)
    
    if flipit:
        flip = tio.RandomFlip(axes = (0), flip_probability = 1.)
        transform_preprocess = tio.Compose([transform_preprocess, flip])
        if label == 1:
            label = 2
        elif label == 2:
            label = 1
        elif label == 3:
            label = 4
        elif label == 4:
            label = 3
    
    if train:
        transforms_spatial = {tio.transforms.RandomAffine(scales = 0.2, degrees = (10,10,10)): 1}
        transform_spatial = tio.OneOf(transforms_spatial, p = .8)
        
        transform_preprocess = tio.Compose([transform_preprocess, transform_spatial])
    

    
    subjects_dataset = tio.SubjectsDataset(subject_list, transform = transform_preprocess)
    
    sampler = labelsampler(patch_size = patch_size, label = label, prob_positive = prob_positive)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])
    logger.info(f'Number of workers {nw}')
    patches_queue = tio.Queue(
        subjects_dataset,
        queue_length,
        samples_per_volume,
        sampler,
        num_workers = nw,
        shuffle_subjects = True,
        shuffle_patches = True,
        start_background=True,
        )

    patches_loader = DataLoader(
        patches_queue,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
        )
    return patches_loader