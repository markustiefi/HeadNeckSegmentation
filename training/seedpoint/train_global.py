# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:25:09 2022

@author: A0067477
"""
import torch

from model import UNet3D
from torch.utils.data import DataLoader

from Trainer_global import Trainer

import os
from utils import get_logger

from dataset_global import dataset_global

from cldice_loss_channels import soft_DiceLoss_channels, soft_dice_cldice_channels

from create_datasets_queue import get_kfold


import platform
import pathlib

pltsys = platform.system()
if pltsys == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

from sklearn.model_selection import KFold


logger = get_logger('UNet3DTrainer')

def train_global(in_channels = 4, out_channels=2, f_maps=32, final_sigmoid =True, 
         num_levels=4, batch_size = 4,
         max_num_epochs=500, foldername ='data/downsampled',
         pretrained = False, transform_dict = None,
         alpha = 0.7, checkpoint_dir = 'weights', artery = 'carotis', fold = 0):
    
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    if in_channels == 1:
        gauss_versions = False
    else:
        gauss_versions = True
      
    
    logger.info('Main begins')
    logger.info(f'Input Channels: {in_channels}')
    logger.info(f'Epochs: {max_num_epochs}')
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info('Device is ' +str(device))
    
    model = UNet3D(in_channels, out_channels, f_maps=f_maps, num_levels = num_levels, final_sigmoid = final_sigmoid)
    if pretrained == True:
        logger.info('Using pretrained weights.')
        if device == 'cuda':
            pretrained_model = torch.load('weights_downsampled/4in_channels_' + str(artery) + '.pytorch')
        else:
            if artery == 'carotis':
                pretrained_model = torch.load('weights_downsampled/4in_channels_carotid_48_64.pytorch', map_location=torch.device('cpu'))
            else:
                pretrained_model = torch.load('weights_downsampled/4in_channels_vertibral_48_64.pytorch', map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_model['model_state_dict'])
    else:
        logger.info('Starting from scratch')
            
    model = model.to(device)
    
    logger.info(model.__class__.__name__)
    logger.info(f'Starting feature maps {f_maps}. Network depth is {num_levels}')
    logger.info(f'Batch size: {batch_size}.')
    logger.info(f'Data augmentations: {transform_dict}')
    
    foldername_train = os.path.join(foldername, 'train') 
    patlist = os.listdir(foldername_train)
    
    logger.info(f'Fold: {fold}')
    train, test = get_kfold(patlist, random_state = 1, n_splits = 5, fold = fold)
    
    filelist_train = [os.path.join(foldername_train, patlist[i]) for i in train]
    logger.info(f'Folder train: {foldername_train} with length {len(filelist_train)}')
    train_set = dataset_global(filelist_train, transform_dict = transform_dict, gauss_versions=gauss_versions, 
                               artery=artery, in_channels = in_channels, train = True)
    
    train_dataloader = DataLoader(train_set, 
                                  batch_size = batch_size, shuffle =True, num_workers = 4)
    
    filelist_eval = [os.path.join(foldername_train, patlist[i]) for i in test]
    logger.info(f'Folder eval: {foldername_train} with length {len(filelist_eval)}')
    eval_set = dataset_global(filelist_eval, transform_dict = None, gauss_versions=gauss_versions, 
                              artery=artery, in_channels = in_channels, train = False)
    eval_dataloader = DataLoader(eval_set, batch_size = 4,  shuffle = True,
                                 num_workers = 4)
    
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    
    loss_func_train = soft_DiceLoss_channels(ignore_background = False)
    loss_func_eval = soft_DiceLoss_channels(ignore_background = False)
    
    logger.info('Using soft_dice_cldice_bce for training.')
    logger.info('Using soft dice loss for evaluation.')
    

    trainer = Trainer(model, train_dataloader, eval_dataloader, optimizer, 
                      loss_func_train, loss_func_eval, 
                      device, max_num_epochs = max_num_epochs, 
                      checkpoint_dir = checkpoint_dir)
    trainer.fit()
    
    

if __name__ == '__main__':
    print('logger activated')

