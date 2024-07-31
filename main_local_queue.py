# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:25:39 2023

@author: q117mt
"""

from training.local.train_local_queue import train_local
import warnings

warnings.filterwarnings('ignore')


fold = 4
if __name__ == '__main__':
    train_local(in_channels = 1, out_channels = 1, f_maps = 32, final_sigmoid = True,
                        num_levels = 4, batch_size = 6, patch_size = 64, max_num_epochs = 30,
                        label = 1, alpha = 0.7, checkpoint_dir = 'weights_local_queue_v2_64_flipped_fold'+ str(fold) + '_',
                        queue_length = 500, samples_per_volume = 100, prob_positive = 9, 
                        path_to_data = r'C:\Users\q117mt\ArterySegmentation\imagesTR', 
                        path_to_labels = r'C:\Users\q117mt\ArterySegmentation\labelsTR',
                        fold = fold, device = 'cuda', flipit = True)

    train_local(in_channels = 1, out_channels = 1, f_maps = 32, final_sigmoid = True,
                        num_levels = 4, batch_size = 6, patch_size = 64, max_num_epochs = 30,
                        label = 2, alpha = 0.7, checkpoint_dir = 'weights_local_queue_v2_64_flipped_fold'+ str(fold) + '_',
                        queue_length = 500, samples_per_volume = 100, prob_positive = 9, 
                        path_to_data = r'C:\Users\q117mt\ArterySegmentation\imagesTR', 
                        path_to_labels = r'C:\Users\q117mt\ArterySegmentation\labelsTR',
                        fold = fold, device = 'cuda', flipit = True)

    train_local(in_channels = 1, out_channels = 1, f_maps = 32, final_sigmoid = True,
                        num_levels = 4, batch_size = 6, patch_size = 64, max_num_epochs = 30,
                        label = 3, alpha = 0.7, checkpoint_dir = 'weights_local_queue_v2_64_flipped_fold'+ str(fold) + '_',
                        queue_length = 500, samples_per_volume = 100, prob_positive = 9, 
                        path_to_data = r'C:\Users\q117mt\ArterySegmentation\imagesTR', 
                        path_to_labels = r'C:\Users\q117mt\ArterySegmentation\labelsTR',
                        fold = fold, device = 'cuda', flipit = True)

    train_local(in_channels = 1, out_channels = 1, f_maps = 32, final_sigmoid = True,
                        num_levels = 4, batch_size = 6, patch_size = 64, max_num_epochs = 30,
                        label = 4, alpha = 0.7, checkpoint_dir = 'weights_local_queue_v2_64_flipped_fold'+ str(fold) + '_',
                        queue_length = 500, samples_per_volume = 100, prob_positive = 9, 
                        path_to_data = r'C:\Users\q117mt\ArterySegmentation\imagesTR', 
                        path_to_labels = r'C:\Users\q117mt\ArterySegmentation\labelsTR',
                        fold = fold, device = 'cuda', flipit = True)

    
