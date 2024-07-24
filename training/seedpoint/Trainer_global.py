# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:11:08 2023

@author: A0067477
"""
from utils import get_logger, RunningAverage, save_checkpoint
from tqdm import tqdm as tq
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

logger = get_logger('Trainer')

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, optimizer, 
                 loss_func_train, loss_func_eval, device, max_num_epochs,
                 checkpoint_dir, eval_score_is_better = True):
    
        self.model = model
        self.train_dataloader = train_dataloader
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
        self.plotit = True
    
    def fit(self):
        
        hist_train = []
        hist_eval = []
        
        for _ in range(self.max_num_epochs):
            
            logger.info(f'Current learning rate: {self.scheduler.get_last_lr()}')
            #should_terminate, loss, [dice_avg, cl_dice_avg, bce_avg] = self.train_tqdm()
            #should_terminate, loss, [dice_avg, bce_avg] = self.train_tqdm()
            should_terminate, loss = self.train_tqdm()
            
            hist_train.append(loss)
            
            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training.')
                return
            
            logger.info(f'Epoch: {self.num_epochs}')
            logger.info(f'Training loss: {loss}')
            #logger.info(f'Training dice loss: {dice_avg}')
            #logger.info(f'Training cl_dice: {cl_dice_avg}')
            #logger.info(f'Training bce_avg: {bce_avg}')
            
            eval_score = self.validate()
            hist_eval.append(eval_score)
            
            is_best = self._is_best_eval_score(eval_score)
            self._save_checkpoint(eval_score, is_best)
            self.num_epochs += 1
            
            #if _ in [20, 40, 60, 80]:
            #    plt.figure(_)
            #    plt.plot(hist_eval, label = 'eval loss', color = 'green')
            #    plt.plot(hist_train, label = 'train loss', color = 'blue')
            #    plt.legend()
            #    plt.title(f'Epoch: {_}')
            #    plt.show()
            
        dic = {'hist_eval': hist_eval, 'hist_train': hist_train}
        np.save(os.path.join(self.checkpoint_dir, 'train_loss'), dic)
        
        plt.figure()
        plt.plot(hist_eval, label = 'eval loss', color = 'green')
        plt.plot(hist_train, label = 'train loss', color = 'blue')
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, 'loss_plot'))
        plt.show()
        logger.info(f'Reached maximum number of epochs: {self.max_num_epochs}. Finishing training.')
        
        
    def train_tqdm(self):
        self.model.train()
        
        train_losses = RunningAverage()
        #dice_avg = RunningAverage()
        #cl_dice_avg = RunningAverage()
        #bce_avg = RunningAverage()
        
        bar = tq(self.train_dataloader, postfix={'train_loss': train_losses.avg}, disable = False)
        
        for inp, target in bar:
            self.optimizer.zero_grad()
            inp = inp.to(device = self.device)
            target = target.to(device = self.device)
            output = self.model(inp)
            
            #loss, [ce, dice] = self.loss_func_train(target, output)
            loss = self.loss_func_train(target, output)
            
            loss.backward()
            self.optimizer.step()
            
            if self.should_stop():
                return True
            
            train_losses.update(loss.item())
            #dice_avg.update(dice.item())
            #cl_dice_avg.update(cl_dice.item())
            #bce_avg.update(ce.item())
            
            bar.set_postfix(ordered_dict = {'train_loss': train_losses.avg})
            bar.update(n=1)
        self.scheduler.step()
        if (self.plotit) & (self.num_epochs in [1,10,25,50,75]):
            in_channels = inp.shape[1]
            if in_channels == 1:
                k = 0
            elif in_channels == 2:
                k = 1
            else:
                k = 2
            for i in range(output.size()[0]):
                plt.figure(2*i, figsize = (9,9))
                plt.subplot(331)
                plt.imshow(np.sum(inp[i][0,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Windowed raw')
                plt.subplot(332)
                plt.imshow(np.sum(inp[i][k,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Gauss 1')
                plt.subplot(333)
                plt.imshow(np.sum(output[i][0].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 0')
                plt.subplot(334)
                plt.imshow(np.sum(target[i][0].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 0')
                plt.subplot(335)
                plt.imshow(np.sum(output[i][1].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 1')
                plt.subplot(336)
                plt.imshow(np.sum(target[i][1].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 1')
                plt.subplot(337)
                plt.imshow(np.sum(output[i][2].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 2')
                plt.subplot(338)
                plt.imshow(np.sum(target[i][2].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 2')
                plt.suptitle(f'Train Carotid,  Epoch {self.num_epochs}')
                plt.show()
                plt.figure((2*i)+1, figsize = (9,9))
                plt.subplot(321)
                plt.imshow(np.sum(inp[i][0,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Windowed raw')
                plt.subplot(322)
                plt.imshow(np.sum(inp[i][k,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Gauss 1')
                plt.subplot(323)
                plt.imshow(np.sum(output[i][3].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 0')
                plt.subplot(324)
                plt.imshow(np.sum(target[i][3].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 0')
                plt.subplot(325)
                plt.imshow(np.sum(output[i][4].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 1')
                plt.subplot(326)
                plt.imshow(np.sum(target[i][4].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 1')
                plt.suptitle(f'Train Vertibral, Epoch {self.num_epochs}')
                plt.show()
        #return False, train_losses.avg, [dice_avg.avg, bce_avg.avg]
        return False, train_losses.avg
    
    def validate(self):
        self.model.eval()
        val_losses = RunningAverage()
        
        with torch.no_grad():
            bar = tq(self.eval_dataloader, postfix = {'eval_loss': val_losses.avg}, disable = False)
            
            for inp, target in bar:
                inp = inp.to(device = self.device)
                target = target.to(device = self.device)
                
                output = self.model(inp)
                
                loss = self.loss_func_eval(target, output)
                
                val_losses.update(loss.item())
                bar.set_postfix(ordered_dict = {'eval_loss': val_losses.avg})
                bar.update(n=1)
                
        if (self.plotit) & (self.num_epochs in [1,10,25,50,75]):
            in_channels = inp.shape[1]
            if in_channels == 1:
                k = 0
            elif in_channels == 2:
                k = 1
            else:
                k = 2
            for i in range(output.size()[0]):
                plt.figure(i, figsize = (9,9))
                plt.subplot(321)
                plt.imshow(np.sum(inp[i][0,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Windowed raw')
                plt.subplot(322)
                plt.imshow(np.sum(inp[i][k,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Gauss 1')
                plt.subplot(323)
                plt.imshow(np.sum(output[i][1].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 0')
                plt.subplot(324)
                plt.imshow(np.sum(target[i][1].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 0')
                plt.subplot(325)
                plt.imshow(np.sum(output[i][2].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 1')
                plt.subplot(326)
                plt.imshow(np.sum(target[i][2].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 1')
                plt.suptitle(f'Evaluation Carotid, Epoch {self.num_epochs}')
                plt.show()
                plt.figure(figsize = (9,9))
                plt.subplot(321)
                plt.imshow(np.sum(inp[i][0,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Windowed raw')
                plt.subplot(322)
                plt.imshow(np.sum(inp[i][k,:,:,:].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Gauss 1')
                plt.subplot(323)
                plt.imshow(np.sum(output[i][3].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 0')
                plt.subplot(324)
                plt.imshow(np.sum(target[i][3].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 0')
                plt.subplot(325)
                plt.imshow(np.sum(output[i][4].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Pred 1')
                plt.subplot(326)
                plt.imshow(np.sum(target[i][4].squeeze().detach().cpu().numpy(), axis = 1))
                plt.title('Groundtruth 1')
                plt.suptitle(f'Evaluation Vertibralis, Epoch {self.num_epochs}')
                plt.show()
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

if __name__ == '__main__':
    print('logger activated')