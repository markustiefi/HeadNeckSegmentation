# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:56:27 2023

@author: A0067477
"""
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
import torch


class dataset_global(Dataset):
    
    def __init__(self, fileslist, transform_dict = None, gauss_versions = False, 
                 artery='carotis', in_channels = 1, train = True):
        self.filelist = fileslist
        self.transform_dict = transform_dict
        self.gauss_versions = gauss_versions
        self.artery = artery
        self.in_channels = in_channels
        self.train = train
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        filepath = self.filelist[idx] 
        subject = np.load(filepath, allow_pickle = True).item()
        
        if self.train:
            if np.random.random()<0.5:
                flip = tio.RandomFlip(axes = (0), flip_probability=1.)
                subject = flip(subject)
                subject.labels.data = torch.stack((subject.labels.data[0], subject.labels.data[2],
                                                   subject.labels.data[1], subject.labels.data[4],
                                                   subject.labels.data[3]), dim = 0)
        
        if self.transform_dict:
            #do spatial transforms.
            if np.random.random()<0.6:
                if np.random.random() < 0.5:
                    raffine = tio.transforms.RandomAffine(scales = 0.1, degrees = (10,10,10), translation=(15,15,10))
                    subject = raffine(subject)
                else:
                    elastic = tio.transforms.RandomElasticDeformation(num_control_points = 7, max_displacement = 11, locked_borders = 2)
                    subject = elastic(subject)
                    
                #Due to one hot encoding already done, the background label becomes 0 at the borders. Bad solution.
                labels = subject.labels.data
                arteries = labels[1]+labels[2]+labels[3]+labels[4]
                subject.labels.data[0] = torch.ones_like(labels[0])-arteries
            
            #Do voxel intensity transforms
            transform = tio.OneOf(self.transform_dict, p =0.5)
            transformed = transform(subject)
            
            raw = transformed.image.data
            label = transformed.labels.data
            
        else:
            raw = subject.image.data
            label = subject.labels.data
        
        if self.in_channels == 1:
            raw = raw[0].unsqueeze(0)
        elif self.in_channels == 2:
            raw = raw[[0, 2]]
        elif self.in_channels == 3:
            raw = raw[[0,1,3]]
        
        #Delete the background channel
        #if label.shape[0] == 5:
        #    label = label[1::]
        
        if self.artery =='carotis':
            label = label[0:2].float()
        elif self.artery == 'vertibralis':
            label = label[2::].float()
        else:
            label = label.float()
            
        if len(raw.shape) == 3:
            raw = raw.unsqueeze(0)

        return raw, label