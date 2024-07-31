# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:01:26 2022

@author: A0067477
"""
import argparse
import torchio as tio
import os

from training.seedpoint.train_global import train_global
from distutils.util import strtobool

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,  description = 'Parse the training input.')


parser.add_argument('--global_local', type = str, default='global')

parser.add_argument('--in_channels', type=int, default=2)
parser.add_argument('--out_channels', type = int, default=5)
parser.add_argument('--foldername', type = str, default = r'downsampled_calculatedmean_multiplesigma')
parser.add_argument('--checkpoint_dir', type=str, default='weights_downsampled_2C_newdataaug_fold')
parser.add_argument('--artery', type = str, default ='all')
parser.add_argument('--transform_dict', type = dict, default = {tio.transforms.RandomSpike(num_spikes = 1, intensity = (0.5,1)): 0.2,
                                                                tio.transforms.RandomNoise(mean = 0, std = (0, 0.1)): 0.2,
                                                                tio.transforms.RandomBiasField(coefficients = 0.8, order = 3): 0.2,
                                                                tio.transforms.RandomBlur(std = (0.4,0.9)): 0.2,
                                                                tio.transforms.RandomMotion(degrees = 1, translation = 1, num_transforms = 2, image_interpolation='linear'): 0.2,
                                                                })

parser.add_argument('--f_maps', type = int, default = 32)
parser.add_argument('--final_sigmoid', type= lambda x: bool(strtobool(x)), default = False)
parser.add_argument('--num_levels', type = int, default = 4)
parser.add_argument('--batch_size', type = int, default = 4)
parser.add_argument('--max_num_epochs', type = int, default = 100)
parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default = False)
parser.add_argument('--alpha', type=float, default = .3)
#parser.add_argument('--fold', type = int, default = 0)

args = parser.parse_args()

dic = args.__dict__.copy()
dic['transform_dict'] = str(dic['transform_dict'])

#with open('commandline_args.txt', 'w') as f:
#    json.dump(dic, f, indent=2)
supfolder = 'weights_downsampled_newdataaug_fold'
if __name__ == '__main__':
    
    if not os.path.isdir(supfolder):
        os.mkdir(supfolder)
        
    for in_channels in [2,1,3,4]:
        for fold in [0,1,2,3,4]:
            weight_dir = f'weights_downsampled_{in_channels}C_fold{fold}'
            checkpoint_dir = os.path.join(supfolder, weight_dir)
            train_global(in_channels = in_channels, out_channels=args.out_channels, f_maps=args.f_maps, final_sigmoid = args.final_sigmoid, 
                     num_levels= args.num_levels, batch_size = args.batch_size,
                     max_num_epochs= args.max_num_epochs, foldername = args.foldername,
                     pretrained = args.pretrained, transform_dict = args.transform_dict, alpha = args.alpha,
                     checkpoint_dir = checkpoint_dir, artery = args.artery, fold = fold)
            


