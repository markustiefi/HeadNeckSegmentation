import numpy as np
import torch
import torchio as tio
import os

from model.model import UNet3D
from inference.centerline_midpoint import get_new_midpoints_fast
from utils.utils_imageclass import getLargestCC, get_start_point_list_fast, threshold_img, get_patch, get_global_index, create_one_hot
from inference.Downsample_class import Downsample
from utils.utils import get_logger

import asyncio
import nest_asyncio
nest_asyncio.apply()



logger = get_logger('Tracking class')
class Image_predict(Downsample):

    def __init__(self, pat: str,
                 path_to_raw: str,
                 gauss_mus: list, gauss_sigma: list,
                 shape: tuple, device: str,
                 path_to_global: str, path_to_local_l: str, path_to_local_r: str, 
                 path_to_nets: str, global_thr: float, safety_break: int,
                 instancenorm: bool, flip_prediction: bool, global_aware_pathfinder: bool,
                 use_direction_momentum: bool, out_channels_local: int,
                 in_channels_local: int, labels = list):
        
        
        super().__init__(pat, path_to_raw, gauss_mus, gauss_sigma, device, shape)
        
        self.path_to_global = path_to_global
        self.path_to_local_l = path_to_local_l
        self.path_to_local_r = path_to_local_r
        self.path_to_nets = path_to_nets
        self.global_thr = global_thr
        self.safety_break = safety_break
        self.instancenorm = instancenorm
        self.flip_prediction = flip_prediction
        self.global_aware_pathfinder = global_aware_pathfinder
        self.use_direction_momentum = use_direction_momentum
        self.is_below = [False, False, False, False]
        self.is_above = [False, False, False, False]
        self.z_min = [0,0,0,0]
        self.z_max = [0,0,0,0]
        self.k = [0,0,0,0]
        self.out_channels_local = out_channels_local
        self.in_channels_local = in_channels_local
        self.labels = labels
        
        if self.out_channels_local > 1:
            self.final_sigmoid = False
        else:
            self.final_sigmoid = True
        
    def get_model(self, path_to_net, f_maps):
        model = UNet3D(in_channels = self.in_channels_local, out_channels = self.out_channels_local, f_maps=f_maps, num_levels = self.num_levels, 
                           instancenorm = self.instancenorm, final_sigmoid = self.final_sigmoid)

        net = os.path.join(path_to_net, 'best_checkpoint.pytorch')
        checkpoint = torch.load(net, map_location=torch.device(self.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def segnfollow_v2(self, subject = None, patch_size = 32, radius = 8, 
                      thr_pathfinder_c = 0.5, thr_pathfinder_v = 0.5, plot_it = False, thr_overall = 0.7):
        if patch_size[-1] != 32:
            self.num_levels = 5
        else:
            self.num_levels = 4
            
        if not subject:
            subject, subject_down = self.pred_downsampled(net = self.path_to_global)
        
        if patch_size[-1] == 64:
            clamp_fortracking = tio.Clamp(out_min = 0, out_max = 1000)
            subject.image.data = clamp_fortracking(subject.image.data)
        pred_patch_and_loc = dict()

        if len(self.labels) > 1:
            parallel = True
        else:
            parallel = False

        if parallel:
            loop = asyncio.get_event_loop()
            looper = asyncio.gather(*[self.segarteries_parallel(a, subject, patch_size, radius, thr_pathfinder_c, thr_pathfinder_v, pred_patch_and_loc) for a in self.labels])
            pred_patch_and_loc = loop.run_until_complete(looper)
            return pred_patch_and_loc[0]
        else:
            for a in self.labels:
                pred_patch_and_loc = self.segarteries(a, subject, patch_size, radius, thr_pathfinder_c, thr_pathfinder_v, pred_patch_and_loc)
            return pred_patch_and_loc
    
    def background(f):
        def wrapped(*args, **kwargs):
            return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    
        return wrapped
    
    @background
    def segarteries_parallel(self, a, subject, patch_size, radius, thr_pathfinder_c, thr_pathfinder_v, pred_patch_and_loc):
        #logger.info(f'Start segmenting label {a}')
        path_to_local = self.path_to_nets[f'weights_{a}']
        
        if a == 1:
            b = 2
        elif a == 2:
            b = 1
        elif a == 3:
            b = 4
        else:
            b = 3
        if self.flip_prediction:
            path_to_local_flipped = self.path_to_nets[f'weights_{b}']
        else:
            path_to_local_flipped = None

        start_point_list, _, _ = get_start_point_list_fast(subject.prediction_downsampled.data[a], subject.image.shape, global_thr = self.global_thr)
        #start_point_list, _, _ = get_random_start_points_fast(subject.prediction_downsampled.data[a], subject.image.shape, global_thr = self.global_thr, n = 5)
        
        
        prediction_thr = np.uint(getLargestCC(threshold_img(np.asarray(subject.prediction_downsampled.data[a].clone()), self.global_thr), allow_break=True))
        self.z_min[a-1] = int(np.min(np.nonzero(prediction_thr)[2])/subject.prediction_downsampled.shape[-1]*subject.image.shape[-1])
        self.z_max[a-1] = int(np.max(np.nonzero(prediction_thr)[2])/subject.prediction_downsampled.shape[-1]*subject.image.shape[-1])
        #logger.info(f'Global prediciton in: [{self.z_min}, {self.z_max}]')
        del prediction_thr
        
        
        if a < 3:
            i = 1
            threshold = thr_pathfinder_c
            self.threshold = threshold
        else:
            i = 3
            threshold = thr_pathfinder_v
            self.threshold = threshold
        
        pred_patch_and_loc_artery = []
        
        for j, start_point in enumerate(start_point_list):

            pred_patch_and_loc_tmp = self.seg_all_v2(subject, start_point,
                                                     nets = path_to_local, a = a, patch_size = patch_size,
                                                     threshold = threshold, radius = radius, plot_it = False, path_to_local_flipped = path_to_local_flipped)
            
            pred_patch_and_loc_artery.append(pred_patch_and_loc_tmp)
            
            del pred_patch_and_loc_tmp
            if self.k[a-1] > 35:
                logger.info(f'No need for second startpoint artery {a}')
                break
        

    
        pred_patch_and_loc.update({f'Label_{a}': pred_patch_and_loc_artery})
        del pred_patch_and_loc_artery
        return pred_patch_and_loc
    
    def segarteries(self, a, subject, patch_size, radius, thr_pathfinder_c, thr_pathfinder_v, pred_patch_and_loc):
        #logger.info(f'Start segmenting label {a}')
        path_to_local = self.path_to_nets[f'weights_{a}']
        
        if a == 1:
            b = 2
        elif a == 2:
            b = 1
        elif a == 3:
            b = 4
        else:
            b = 3
        if self.flip_prediction:
            path_to_local_flipped = self.path_to_nets[f'weights_{b}']
        else:
            path_to_local_flipped = None
        

        start_point_list, _, _ = get_start_point_list_fast(subject.prediction_downsampled.data[a], subject.image.shape, global_thr = self.global_thr)
        #start_point_list, _, _ = get_random_start_points_fast(subject.prediction_downsampled.data[a], subject.image.shape, global_thr = self.global_thr, n = 5)
        
        
        prediction_thr = np.uint(getLargestCC(threshold_img(np.asarray(subject.prediction_downsampled.data[a].clone()), self.global_thr), allow_break=True))
        self.z_min[a-1] = int(np.min(np.nonzero(prediction_thr)[2])/subject.prediction_downsampled.shape[-1]*subject.image.shape[-1])
        self.z_max[a-1] = int(np.max(np.nonzero(prediction_thr)[2])/subject.prediction_downsampled.shape[-1]*subject.image.shape[-1])
        #logger.info(f'Global prediciton in: [{self.z_min}, {self.z_max}]')
        del prediction_thr
        
        
        if a < 3:
            i = 1
            threshold = thr_pathfinder_c
            self.threshold = threshold
        else:
            i = 3
            threshold = thr_pathfinder_v
            self.threshold = threshold
        
        pred_patch_and_loc_artery = []
        
        for j, start_point in enumerate(start_point_list):

            pred_patch_and_loc_tmp = self.seg_all_v2(subject, start_point,
                                                     nets = path_to_local, a = a, patch_size = patch_size,
                                                     threshold = threshold, radius = radius, plot_it = False, path_to_local_flipped = path_to_local_flipped)
            
            pred_patch_and_loc_artery.append(pred_patch_and_loc_tmp)
            del pred_patch_and_loc_tmp
            if self.k[a-1] > 35:
                logger.info(f'No need for second startpoint artery {a}')
                break

    
        pred_patch_and_loc.update({f'Label_{a}': pred_patch_and_loc_artery})
        del pred_patch_and_loc_artery
        return pred_patch_and_loc

    def seg_all_v2(self, subject, start_point, nets, a,
                patch_size = 32, radius = 5, threshold = 0.5, plot_it = False,
                path_to_local_flipped = None):
        
        raw = subject.image.data[0]
        
        pred_patch_and_loc = []
            
        z_center, y_center, x_center = start_point
        
        patch = get_patch(raw, z_center, y_center, x_center, patch_size = patch_size, for_network = True)
        
        if not (patch.size()[-1] == patch_size[-1]*2) or not (patch.size()[-2] == patch_size[-2]*2) or not (patch.size()[-3] == patch_size[-3]*2):
            logger.info('Extracted patch has different size. Break at Startpoint location.')
            return pred_patch_and_loc

        pred_sq, prediction = self.seg_patch(patch, nets, threshold = threshold, path_to_local_flipped = path_to_local_flipped, a = a)
        
        
        if len(np.unique(pred_sq)) == 1:
            logger.info('Bad starting point, skip and try next.')
            pred_patch_and_loc.append(tuple((torch.zeros_like(prediction), start_point, 0, 
                                                    self.is_below, self.is_above, torch.zeros_like(prediction))))
            return pred_patch_and_loc
        
        
        direction_1, direction_2, sphere, centerline, projected_mid = get_new_midpoints_fast(pred_sq, radius = radius, 
                                                                                        patch_size = patch_size)
        
        
        
        if np.linalg.norm(direction_1-direction_2)<(radius*np.pi*15/180):
            logger.info('No start directions found, skip and try next.')
            pred_patch_and_loc.append(tuple((torch.zeros_like(prediction), start_point, 0, 
                                                    self.is_below, self.is_above, torch.zeros_like(prediction))))
            return pred_patch_and_loc
        
        pred_patch_and_loc.append(tuple((prediction, start_point, 0, self.is_below, self.is_above, centerline)))
        
        
        start_1 = get_global_index(direction_1, start_point, patch_size)
        start_2 = get_global_index(direction_2, start_point, patch_size)
            
        projected_mid = get_global_index(projected_mid, start_point, patch_size)
        self.k[a-1] = 0
        pred_patch_and_loc = self.seg_direction_v2(raw,
                                                   start_point, start_1, start_2,
                                                   patch_size, nets, pred_patch_and_loc, 
                                                   globlabel = subject.prediction_upsampled.data, a = a,
                                                   radius=radius,threshold=threshold, 
                                                   path_to_local_flipped = path_to_local_flipped)
        
        pred_patch_and_loc = self.seg_direction_v2(raw,
                                                   start_point, start_2, start_1,
                                                   patch_size, nets, pred_patch_and_loc,
                                                   globlabel = subject.prediction_upsampled.data, a = a,
                                                   radius = radius, threshold=threshold,
                                                   path_to_local_flipped = path_to_local_flipped)
            
        return pred_patch_and_loc
    
    def seg_direction_v2(self, raw, start_point, 
                  direction_1, direction_2, patch_size, nets, pred_patch_and_loc,
                  globlabel, a, radius = 5, threshold = 0.5, plot = False,
                  path_to_local_flipped = None):
        
        #Safety measure for vertibral artery such that we do not continue to segment opposite artery.
        if a == 3:
            opposite_artery = 4
            globlabel = getLargestCC(threshold_img(globlabel[opposite_artery].numpy(), self.global_thr), allow_break=True)
        elif a == 4:
            opposite_artery = 3
            globlabel = getLargestCC(threshold_img(globlabel[opposite_artery].numpy(), self.global_thr), allow_break=True)
        
        centers = [start_point, direction_1]
        center = direction_1
        
        momentum_intersection = 0
        momentum_direction = 0
        self.momentum_global = [0,0,0,0]
        while np.linalg.norm(direction_1-direction_2)>(radius*np.pi*15/180):
            x_center, y_center, z_center = center
            patch = get_patch(raw, x_center, y_center, z_center, patch_size, for_network = True)
            if z_center-32 < self.z_min[a-1]:
                self.is_below[a-1] = True
                self.momentum_global[a-1] += 1
            else:
                self.is_below[a-1] = False
                self.momentum_global[a-1] = 0
            if z_center+32 > self.z_max[a-1]:
                self.is_above[a-1] = True
                self.momentum_global[a-1] += 1
            else:
                self.is_above[a-1] = False
                self.momentum_global[a-1] = 0

            if not (patch.size()[-1] == patch_size[-1]*2) or not (patch.size()[-2] == patch_size[-2]*2) or not (patch.size()[-3] == patch_size[-3]*2):
                logger.info('Extracted patch has different size. break')
                break
            pred_sq, prediction = self.seg_patch(patch, nets, threshold = threshold, path_to_local_flipped = path_to_local_flipped, a = a)
            

        
            if len(np.unique(pred_sq)) == 1:
                logger.info('Only one unique value in patch for tracking -> break after {k[a-1]} iterations')
                break
            
            direction_1, direction_2, sphere, centerline, projected_mid = get_new_midpoints_fast(pred_sq, 
                                                                                            radius = radius, 
                                                                                            patch_size = patch_size)
            
            if np.linalg.norm(direction_1-direction_2)<(radius*np.pi*15/180):
                direction_1, direction_2, sphere, centerline, projected_mid = get_new_midpoints_fast(pred_sq, 
                                                                                                radius = radius/2, 
                                                                                                patch_size = patch_size)
                #print(f'Used radius 1/2 for {a}')
            if np.linalg.norm(direction_1-direction_2)<(radius*np.pi*15/180):
                direction_1, direction_2, sphere, centerline, projected_mid = get_new_midpoints_fast(pred_sq, 
                                                                                                radius = radius/4, 
                                                                                                patch_size = patch_size)
                
                #print(f'Used radius 1/4 for {a}')
            if radius > 16:
                if np.linalg.norm(direction_1-direction_2)<(radius*np.pi*15/180):
                    direction_1, direction_2, sphere, centerline, projected_mid = get_new_midpoints_fast(pred_sq, 
                                                                                                    radius = radius/8, 
                                                                                                    patch_size = patch_size)
                    
                    #print(f'Used radius 1/8 for {a}')
            #if np.linalg.norm(direction_1-direction_2)<(radius*np.pi*15/180):
                #print(f'Stopped at {z_center} for {a}')
            
            pred_patch_and_loc.append(tuple((prediction, center, 0, self.is_below[a-1], self.is_above[a-1], centerline)))
            
            
            center, old_direction = self.direction_decider(direction_1, direction_2, patch_size, centers)
            

            if self.use_direction_momentum:
                norm_direction_momentum = np.linalg.norm(center-centers, axis = 1)
                if np.min(norm_direction_momentum) < radius:
                    if momentum_direction > 3:
                        #logger.info(f'Exiting since the center was already seen after {k[a-1]} iterations')
                        break
                    else:
                        momentum_direction += 1
                else:
                    momentum_direction = 0
            
            centers.append(center)
            
            #if np.linalg.norm(direction_1-direction_2)<(radius*np.pi*15/180):
                #logger.info(f'Exiting loop due to while condition after {k[a-1]} iterations.')
            
            if self.k[a-1] > self.safety_break:
                logger.info(f'Safety break used after {self.k[a-1]} iterations')
                break
            
            if a >= 3:
                patch_glob = get_patch(globlabel, x_center, y_center, z_center, patch_size, for_network = False)
                intersection = patch_glob*pred_sq
                if np.count_nonzero(intersection) != 0:
                    if momentum_intersection > 2:
                        #logger.info(f'Intersection with global prediciton of other artery after {k[a-1]} iterations')
                        break
                    #logger.info('Intersection with global prediciton of other artery, but using momentum.')
                    momentum_intersection += 1
            self.k[a-1] += 1
        return pred_patch_and_loc
        
    def seg_patch(self, patch, nets, threshold = 0.7, path_to_local_flipped = None, a = 1):
        patch = patch.to(self.device)
        prediction = None
        for net in nets:
            model = self.get_model(path_to_net = net, f_maps = 32)
            model = model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                if prediction is None:
                    prediction = model(patch.float())
                else:
                    prediction += model(patch.float())
        len_nets = len(nets)
        if path_to_local_flipped is not None:
            prediction_flipped = None
            if not self.device == 'cpu':
                patch = patch.cpu()
            patch = tio.Subject(image = tio.ScalarImage(tensor = patch[0]))
            flip = tio.RandomFlip(axes = (0), flip_probability=1.)
            patch = flip(patch)
            patch = patch.image.data.unsqueeze(0).to(self.device)
            
            for net in path_to_local_flipped:
                model = self.get_model(path_to_net = net, f_maps = 32)
                model = model.to(self.device)
                model.eval()
                
                with torch.no_grad():
                    if prediction_flipped is None:

                        prediction_flipped = model(patch.float())
                    else:
                        prediction_flipped += model(patch.float())
            if not self.device == 'cpu':
                prediction_flipped = prediction_flipped[0].cpu()
            else:
                prediction_flipped = prediction_flipped[0]
                
            prediction_flipped = tio.Subject(image = tio.ScalarImage(tensor = prediction_flipped))
            prediction_flipped = flip(prediction_flipped)
            prediction = prediction+prediction_flipped.image.data.unsqueeze(0).to(self.device)
            len_nets += len(path_to_local_flipped)
            
        if not self.device == 'cpu':
            prediction = prediction.cpu()/len_nets
        else:
            prediction = prediction/len_nets
        pred_sq = prediction.squeeze().numpy().copy()
        if self.global_aware_pathfinder:
            if (self.is_above[a-1]) or (self.is_below[a-1]): 
                logger.info(f'Use higher pathfinder threshold, because is_above is {self.is_above} or is_below is {self.is_below}')
                if self.momentum_global[a-1] > 3:
                    threshold = 0.5
            else:
                threshold = self.threshold
                
        logger.info(f'Threshold after checking whether the current patch is above or below global prediction: {threshold}' )
        
        pred_sq = threshold_img(pred_sq, threshold=threshold)
        if pred_sq.shape[0] == 3:
            pred_sq = pred_sq[1]+pred_sq[2]
        if len(np.unique(pred_sq)) == 1:
            logger.info('Only one unique value in patch for tracking')
            pred_sq = np.zeros_like(pred_sq)
        else:
            pred_sq = getLargestCC(np.array(pred_sq), allow_break=True)
        return pred_sq, prediction
    
        
    def direction_decider(self, direction_1, direction_2, patch_size, centers):
        
        direction_1_tmp = get_global_index(direction_1, centers[-1], patch_size)
        direction_2_tmp = get_global_index(direction_2, centers[-1], patch_size)
        
        norm1 = np.linalg.norm(direction_1_tmp-centers[-2])
        norm2 = np.linalg.norm(direction_2_tmp-centers[-2])
        
        if norm2<norm1:
            return direction_1_tmp, direction_2_tmp
        else:
            return direction_2_tmp, direction_1_tmp        
    
    
        
        
        
