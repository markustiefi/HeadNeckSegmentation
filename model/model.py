# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:31:19 2022

@author: A0067477
"""

import torch.nn as nn

from model.buildingblocks import DoubleConv, create_encoders, create_decoders
from utils.utils import number_of_features_per_level


class Abstract3DUNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module,
                 f_maps = 64, num_levels=4, conv_kernel_size = 3, pool_kernel_size = 2, instancenorm = True,
                 conv_padding = 1, **kwargs):
        super(Abstract3DUNet, self).__init__()
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels = num_levels)
            
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps)>1
        
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, pool_kernel_size, instancenorm = instancenorm)
        
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, instancenorm = instancenorm)
        
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim = 1)
            
    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0,x)
            
        encoders_features = encoders_features[1:]
        
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
            
        x = self.final_conv(x)
        x = self.final_activation(x)
        #if not self.training and self.final_activation is not None:
        #    x = self.final_activation(x)
        return x
    
class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid = True, f_maps = 64, num_levels=4, instancenorm = True,
                 conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels = out_channels,
                                     final_sigmoid = final_sigmoid,
                                     basic_module = DoubleConv,
                                     f_maps = f_maps,
                                     num_levels = num_levels,
                                     instancenorm = instancenorm,
                                     conv_padding = conv_padding,
                                     **kwargs)
        self.in_channels = in_channels



class UNet2D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module,
                 f_maps = 32, num_levels = 4, conv_kernel_size = 3, pool_kernel_size = 2,
                 conv_padding = 1):
        if conv_padding ==1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels = out_channels,
                                     final_sigmoid = final_sigmoid,
                                     basic_module = DoubleConv,
                                     f_maps = f_maps,
                                     num_levels = num_levels,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding)
    
            
            
        



#def get_model(model_config):
#    model_class = get_class(model_config['name'], modules=['model'])
#    return model_class(**model_config)















