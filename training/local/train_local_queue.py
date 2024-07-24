# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:19:46 2023

@author: q117mt
"""


import torch

from model.model import UNet3D
from datahandling.create_kfold import get_kfold
from training.local.dataloader_queue import Queue
from training.local.Trainer_local_queue import Trainer
from model.cldice_loss import soft_DiceLoss, soft_dice_cldice_bce

import os
from utils.utils import get_logger


logger = get_logger('UNet3DTioQueue')

def train_local(in_channels, out_channels, f_maps, final_sigmoid,
                num_levels, batch_size, patch_size, max_num_epochs,
                label, alpha, checkpoint_dir,
                queue_length = 50, samples_per_volume = 100, prob_positive = 8, 
                path_to_data = 'imagesTr', path_to_labels = 'labelsTr',
                fold = 0, device = 'cuda', flipit = True):
    
    logger.info('Train local begins.')
    logger.info(f'Input channels: {in_channels}')
    logger.info(f'Epochs: {max_num_epochs}')
    
    checkpoint_dir = checkpoint_dir + 'label_' + str(label)
    
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    
    logger.info(f'Device is {device}' )
    model = UNet3D(in_channels=in_channels, out_channels = out_channels,
                   f_maps = f_maps, num_levels = num_levels)
    
    logger.info('Starting from scratch')
    
    model = model.to(device)
    
    logger.info(model.__class__.__name__)
    logger.info(f'Feature maps: {f_maps}. Number of levels: {num_levels}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Patch size: {patch_size}')
    
    patlist_raw = os.listdir(path_to_data)
    patlist_label = os.listdir(path_to_labels)
    logger.info(f'Fold: {fold}')
    train, test = get_kfold(patlist_raw, random_state = 1, n_splits = 5, fold = fold)
    
    filelist_train_raw = [os.path.join(path_to_data, patlist_raw[i]) for i in train]
    filelist_train_labels = [os.path.join(path_to_labels, patlist_label[i]) for i in train]
    
    filelist_eval_raw = [os.path.join(path_to_data, patlist_raw[i]) for i in test]
    filelist_eval_labels = [os.path.join(path_to_labels, patlist_label[i]) for i in test]
    
    logger.info(f'Training on {len(filelist_train_raw)} volumes.')
    logger.info(f'Evaluating on {len(filelist_eval_raw)} volumes.')
    
    train_dataloader = Queue(filelist_train_raw, filelist_train_labels, batch_size = batch_size,
                             patch_size=patch_size, label = label, queue_length=queue_length,
                             samples_per_volume=samples_per_volume, train = True, prob_positive=prob_positive)
    if flipit:
        train_dataloader_flipped = Queue(filelist_train_raw, filelist_train_labels, batch_size = batch_size,
                                 patch_size=patch_size, label = label, queue_length=queue_length,
                                 samples_per_volume=samples_per_volume, train = True, prob_positive=prob_positive, flipit = True)
    else:
        train_dataloader_flipped = None
    
    eval_dataloader = Queue(filelist_eval_raw, filelist_eval_labels, batch_size = batch_size,
                             patch_size=patch_size, label = label, queue_length=queue_length,
                             samples_per_volume=samples_per_volume, train = False, prob_positive=prob_positive)
    
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr = 0.02)
    
    loss_func_train = soft_dice_cldice_bce(alpha = alpha, dice_alpha=0.3, smooth = 1.)
    loss_func_eval = soft_DiceLoss(smooth = 1.)
    
    logger.info('Using soft dice cldice bce for training.')
    logger.info('Using soft dice loss for evaluation.')
    
    trainer = Trainer(model, label, train_dataloader, train_dataloader_flipped, eval_dataloader, optimizer, 
                      loss_func_train, loss_func_eval, 
                      device, max_num_epochs, checkpoint_dir)
    
    trainer.fit()
    return
    
    
    