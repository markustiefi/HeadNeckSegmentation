# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:39:50 2022

@author: A0067477
"""

import torch
from torch import nn as nn
from torch.nn import functional as F

from functools import partial

def conv3d(in_channels, out_channels, kernel_size, padding, bias = True):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding = padding, bias = bias)

def create_conv(in_channels, out_channels, kernel_size, padding, instancenorm=True):
    modules = []
    modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, padding=padding)))
    modules.append(('ReLU', nn.ReLU(inplace=True)))
    if instancenorm:
        modules.append(('instancenorm', nn.InstanceNorm3d(out_channels, affine = True)))
    else:
        modules.append(('batchnorm', nn.BatchNorm3d(out_channels, affine = True))) 
    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding=1, instancenorm=True):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, padding, instancenorm):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, padding=1, instancenorm=True):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv2_out_channels, 
                                   kernel_size, padding=padding, instancenorm=instancenorm))
        
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, 
                                   kernel_size, padding=padding, instancenorm=instancenorm))
        
        
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size = 3, apply_pooling = True,
                 pool_kernel_size = 2, pool_type = 'max', basic_module = DoubleConv, padding = 1,
                 instancenorm=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size = pool_kernel_size)
        else:
            self.pooling = None
        
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder = True,
                                         kernel_size = conv_kernel_size,
                                         padding = padding,
                                         instancenorm = instancenorm)
        if out_channels > 65:
            self.dropout = torch.nn.Dropout(p=0.33)
        else:
            self.dropout = None
        
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size = 3, 
                 scale_factor = (2,2,2), basic_module = DoubleConv, mode = 'nearest',
                 padding = 1, upsample = True, instancenorm = True):
        super(Decoder, self).__init__()
        
       
        self.upsampling = InterpolateUpsampling(mode=mode)
        self.joining = partial(self._joining, concat = True)
        
        self.basic_module = basic_module(in_channels, out_channels, 
                                         encoder=False, kernel_size= conv_kernel_size,
                                         padding = padding, instancenorm = instancenorm)
    
    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features = encoder_features, x = x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x
    
    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features+x
        

def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, 
                    conv_padding, pool_kernel_size, instancenorm = True):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,
                              basic_module=basic_module,
                              conv_kernel_size = conv_kernel_size,
                              padding = conv_padding,
                              instancenorm = instancenorm)
        else:
            encoder = Encoder(f_maps[i-1], out_feature_num,
                              basic_module = basic_module,
                              conv_kernel_size = conv_kernel_size,
                              pool_kernel_size = pool_kernel_size, 
                              padding = conv_padding,
                              instancenorm = instancenorm)
    
        encoders.append(encoder)
    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, 
                    upsample = True, instancenorm = True):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps)-1):
        in_feature_num = reversed_f_maps[i]+reversed_f_maps[i + 1]
        out_feature_num = reversed_f_maps[i + 1]
        
        _upsample = True
        if i == 0:
            _upsample = upsample
        
        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module = basic_module,
                          conv_kernel_size = conv_kernel_size,
                          padding = conv_padding,
                          instancenorm = instancenorm)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)        
   
class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
