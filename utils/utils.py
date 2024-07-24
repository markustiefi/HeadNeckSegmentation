# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:48:00 2022

@author: A0067477
"""
#import importlib
import logging
import sys
import os
import torch
import shutil
import numpy as np
import torch
from torch import optim


def save_checkpoint(state, checkpoint_dir, is_best = False):
    
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)
        
def save_metadate(dictionary, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    file_name = os.path.join(checkpoint_dir, 'metadata.npy')
    np.save(dictionary, file_name)

loggers = {}
def get_logger(name, level=logging.INFO, formatter = '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s'):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger
        return logger

class RunningAverage:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, value, n=1):
        self.count += n
        self.sum += value
        self.avg = self.sum/self.count 

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** n for n in range(num_levels)]


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer

def padsize(kernel_size=3, mode="same", dilation=1):
    """
    translates mode to size of padding
    """
    if not isinstance(kernel_size, list):
        k = [kernel_size, kernel_size]
    else:
        k = kernel_size
    if not isinstance(dilation, list):
        d = [dilation, dilation]
    else:
        d = dilation
    assert len(d) == len(k)

    p = [0 for _ in range(len(k))]
    if mode == "same":
        for i in range(len(p)):
            p[i] = (d[i] * (k[i] - 1)) // 2

    if np.unique(p).shape[0] == 1:
        p = p[0]
    return p

def get_activation(activation="prelu", **kwargs):
    if activation == "prelu":
        num_parameters = kwargs.get("activation_num_parameters", 1)
        init = kwargs.get("activation_init", 0.25)
        return torch.nn.PReLU(num_parameters, init=init)
    elif activation == "identity":
        return torch.nn.Identity()
    elif activation == "softmax":
        return torch.nn.Softmax(dim=1)
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        return torch.nn.ReLU(inplace=kwargs.get("activation_inplace", False))





