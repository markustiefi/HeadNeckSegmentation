# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:00:22 2022

@author: A0067477
"""
import torch
import torch.nn as nn
from model.soft_skeleton import soft_skel

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        
    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        #flatten label and prediciton tensors
        bs = y_pred.size(0)
        p = y_pred.reshape(bs, -1)
        t = y_true.reshape(bs, -1)
        p_skel = skel_pred.reshape(bs, -1)
        t_skel = skel_true.reshape(bs, -1)
        
        tprec = ((p_skel*t).sum(1)+self.smooth)/(p_skel.sum(1)+self.smooth)
        tsens = ((t_skel*p).sum(1)+self.smooth)/(t_skel.sum(1)+self.smooth)
        cl_dice = 1.-2.*(tprec*tsens)/(tprec+tsens)
        return cl_dice.mean()

class soft_DiceLoss(nn.Module):
    def __init__(self, smooth= 1.):
        super(soft_DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_true, y_pred):
        #flatten label and prediciton tensors
        bs = y_pred.size(0)
        p = y_pred.reshape(bs, -1)
        t = y_true.reshape(bs, -1)
        
        intersection = (p*t).sum(1)
        total = (t+p).sum(1)
        dice = 1.-((2.*intersection + self.smooth)/(total+self.smooth))
        return dice.mean()
    
class soft_dice_cldice(soft_DiceLoss, soft_cldice):
    def __init__(self, iter_=12, alpha=0.5, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        dice = soft_DiceLoss().forward(y_true, y_pred)
        cl_dice = soft_cldice(iter_ = self.iter, smooth =self.smooth).forward(y_true, y_pred)
        return (1.-self.alpha)*dice+self.alpha*cl_dice, dice, cl_dice


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
        
    def forward(self, y_true, y_pred, weighted = False):
        loss = nn.BCELoss(reduction = 'none')
        output = loss(y_pred, y_true)
        output = torch.mean(output)
        return output
    
class soft_dice_cldice_bce(soft_DiceLoss, soft_cldice, BinaryCrossEntropy):
    def __init__(self, alpha = 0.3, dice_alpha = 0.3, smooth=1.):
        super(soft_dice_cldice_bce, self).__init__()
        self.alpha = alpha
        self.dice_alpha = dice_alpha
        self.smooth = smooth
        
    def forward(self, y_true, y_pred):
        bce = BinaryCrossEntropy()(y_true, y_pred)
        soft_dice, dice, cl_dice = soft_dice_cldice(alpha=self.dice_alpha)(y_true, y_pred)
        loss = (1.-self.alpha)*bce+self.alpha*soft_dice
        return loss, [dice, cl_dice, bce]
        
        
        
        
