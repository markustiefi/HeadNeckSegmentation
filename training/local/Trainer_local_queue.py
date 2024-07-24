# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:20:49 2023

@author: q117mt
"""
from utils.utils import get_logger, RunningAverage, save_checkpoint
from tqdm import tqdm as tq
import os
import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

logger = get_logger('Trainer')

class Trainer:
    def __init__(self, model, label, train_dataloader, train_dataloader_flipped, eval_dataloader, optimizer, 
                 loss_func_train, loss_func_eval, device, max_num_epochs,
                 checkpoint_dir, eval_score_is_better = True):
    
        self.model = model
        self.label = label
        self.train_dataloader = train_dataloader
        self.train_dataloader_flipped = train_dataloader_flipped
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.loss_func_train = loss_func_train
        self.loss_func_eval = loss_func_eval
        self.device = device
        self.max_num_epochs = max_num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = ExponentialLR(self.optimizer, gamma = 0.9)
        
        self.num_iter = 1
        self.num_epochs = 1
        self.eval_score_is_better = eval_score_is_better
        self.best_eval_score = float('+inf')
        self.best_eval_score_after_flipped = float('+inf')
        
        if self.label == 1:
            self.label_flipped = 2
        elif self.label == 2:
            self.label_flipped = 1
        elif self.label == 3:
            self.label_flipped = 4
        elif self.label == 4:
            self.label_flipped = 3
            
    
    def fit(self):
        logger.info('Start Training:')
    
        hist_train = []
        hist_eval = []
        for _ in range(self.max_num_epochs):
            
            should_terminate, loss, [dice_avg, cl_dice_avg, bce_avg] = self.train_tqdm(self.train_dataloader, flipped = False)
            
            hist_train.append(loss)
            
            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training.')
                return
            logger.info(f'Epoch: {self.num_epochs}')
            logger.info(f'Training loss: {loss}')
            logger.info(f'Training dice loss: {dice_avg}')
            logger.info(f'Training cl_dice: {cl_dice_avg}')
            logger.info(f'Training bce_avg: {bce_avg}')

            eval_score = self.validate()
            hist_eval.append(eval_score)
            
            is_best = self._is_best_eval_score(eval_score)
            self._save_checkpoint(eval_score, is_best)
            self.num_epochs += 1
            
        if self.train_dataloader_flipped is not None:
            logger.info(f'Number of epochs: {self.num_epochs}')
            logger.info(f'Start training on flipped data.')
                
            for _ in range(int(self.max_num_epochs/2)):
                
                should_terminate, loss, [dice_avg, cl_dice_avg, bce_avg] = self.train_tqdm(self.train_dataloader_flipped, flipped = True)
                
                hist_train.append(loss)
                
                if should_terminate:
                    logger.info('Stopping criterion is satisfied. Finishing training.')
                    return
                logger.info(f'Epoch: {self.num_epochs}')
                logger.info(f'Training loss: {loss}')
                logger.info(f'Training dice loss: {dice_avg}')
                logger.info(f'Training cl_dice: {cl_dice_avg}')
                logger.info(f'Training bce_avg: {bce_avg}')
    
                eval_score = self.validate()
                hist_eval.append(eval_score)
                
                is_best = self._is_best_eval_score(eval_score)
                self._save_checkpoint(eval_score, is_best)
                self.num_epochs += 1
                
            logger.info(f'Number of epochs: {self.num_epochs}')
            logger.info(f'Start refining on original data.')
                
            for _ in range(int(self.max_num_epochs)):
                
                should_terminate, loss, [dice_avg, cl_dice_avg, bce_avg] = self.train_tqdm(self.train_dataloader, flipped = False)
                
                hist_train.append(loss)
                
                if should_terminate:
                    logger.info('Stopping criterion is satisfied. Finishing training.')
                    return
                logger.info(f'Epoch: {self.num_epochs}')
                logger.info(f'Training loss: {loss}')
                logger.info(f'Training dice loss: {dice_avg}')
                logger.info(f'Training cl_dice: {cl_dice_avg}')
                logger.info(f'Training bce_avg: {bce_avg}')
    
                eval_score = self.validate()
                hist_eval.append(eval_score)
                
                is_best = self._is_best_eval_score(eval_score)
                self._save_checkpoint(eval_score, is_best)
                self.num_epochs += 1
                
        logger.info(f'Number of epochs: {self.num_epochs}') 
        dic = {'hist_eval': hist_eval, 'hist_train': hist_train}
        np.save(os.path.join(self.checkpoint_dir, 'train_loss'), dic)
        plt.figure()
        plt.plot(hist_eval, label = 'Loss Eval (Dice)', color = 'green')
        plt.plot(hist_train, label = 'Loss Train (Dice+clDice+BCE)', color = 'blue')
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, 'loss_plot'))
        plt.show()
        
        logger.info(f'Reached maximum number of epochs: {self.max_num_epochs}. Finishing training.')
        
    @staticmethod
    def intensity_tranforms(inp, target):
        transforms = {tio.transforms.RandomSpike(num_spikes = 3, intensity = (0.05,0.1)): 0.25,
                                tio.transforms.RandomNoise(mean = 0, std = (0.01, 0.08)): 0.25,
                                tio.transforms.RandomBlur(std = (0.2,.8)): 0.25,
                                tio.transforms.RandomGamma(log_gamma = (-0.8,0.4)): 0.25}
        
        
        rescale = tio.RescaleIntensity(out_min_max = (0,1))
        
        inp_out = []
        target_out = []
        for b in range(inp.size(0)):
            subject_tmp = tio.Subject({
            'image': tio.ScalarImage(tensor = inp[b,:]),
            'label': tio.LabelMap(tensor = target[b,:])})
            
            oneof = tio.OneOf(transforms, p= 0.7)
            transform = tio.Compose([oneof, rescale])
            
            subject_tmp = transform(subject_tmp)
            
            inp_out.append(subject_tmp.image.data)
            target_out.append(subject_tmp.label.data)
        
        inp = torch.stack(inp_out, dim = 0)
        target = torch.stack(target_out, dim = 0)
        
        return inp, target
        
    def train_tqdm(self, dataloader, flipped = False):
        self.model.train()
        
        train_losses = RunningAverage()
        dice_avg = RunningAverage()
        cl_dice_avg = RunningAverage()
        bce_avg = RunningAverage()
        
        for i, patches_batch in enumerate(dataloader):
            
            inp = patches_batch['image'][tio.DATA]
            target = patches_batch['label'][tio.DATA]
            
            target = target.float()
            if flipped:
                target[target != self.label_flipped] = 0
            else:
                target[target != self.label] = 0
            target /= torch.tensor([target.max(), 1.]).max()
        
            
            inp, target = self.intensity_tranforms(inp, target)    
            inp = inp.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(inp)
            output = torch.nn.Sigmoid()(output)
            
            loss, [dice, cl_dice, bce] = self.loss_func_train(target, output)
            loss.backward()
            self.optimizer.step()
            
            if self.should_stop():
                return True
            
            train_losses.update(loss.item())
            dice_avg.update(dice.item())
            cl_dice_avg.update(cl_dice.item())
            bce_avg.update(bce.item())

        self.scheduler.step()
        return False, train_losses.avg, [dice_avg.avg, cl_dice_avg.avg, bce_avg.avg]
    
    def validate(self):
        self.model.eval()
        val_losses = RunningAverage()
        logger.info('Evaluate')
        with torch.no_grad():
            for patches_batch in self.eval_dataloader:
                inp = patches_batch['image'][tio.DATA].to(device = self.device)
                target = patches_batch['label'][tio.DATA].to(device = self.device)
                target = target.float()
                target[target != self.label] = 0
                target /= torch.tensor([target.max(), 1]).max()
                if len(np.unique(target.cpu()))>2:
                    print(np.unique(target.cpu()))
                
                
                output = self.model(inp)
                
                loss = self.loss_func_eval(target, output)
                
                val_losses.update(loss.item())
        return val_losses.avg
    
    
    
    def _is_best_eval_score(self, eval_score):
        
        is_best = eval_score<self.best_eval_score
        
        if is_best:
            self.best_eval_score = eval_score
            logger.info(f'Saving new best evaluation metric: {eval_score}')
        return is_best
    
    def should_stop(self):
        return False
    
    @staticmethod
    def _batch_size(inp):
        return inp.size(0)
    
    def _save_checkpoint(self, eval_score, is_best = False):
        state_dict = self.model.state_dict()
        
        if is_best:
            checkpoint_name = 'best_checkpoint.pytorch'
        else:
            checkpoint_name = 'last_checkpoint.pytorch'
            
        last_file_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        logger.info(f'Saving checkpoint with evaluation metric: {eval_score}')
        save_checkpoint({
            'num_epochs': self.num_epochs+1,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict()},
            checkpoint_dir = self.checkpoint_dir, is_best = is_best)