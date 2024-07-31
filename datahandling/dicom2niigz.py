# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:38:22 2023

@author: A0067477
"""

import dicom2nifti
import glob
import os
import nibabel as nib
import numpy as np

#path_to_dicom = r'C:\Users\A0067477\3DArterySegmentation\CTA_oktober\Patient101\10000000\10000001\10000002'

#path_to_save_nifti = r'C:\Users\A0067477\3DArterySegmentation\CTA_oktober\Patient101'

#dicom2nifti.convert_directory(path_to_dicom, path_to_save_nifti)

#path = r'C:\Users\A0067477\3DArterySegmentation\CTA'
#walk_through = r'*/*/*'
#os.listdir(path)

#path_to_nnUnet_raw = r'C:\Users\A0067477\3DArterySegmentation\nnUNet_raw_data_base\nnUnet_raw_data\Task501_Artery\imagesTR'
#path_to_nnUnet_label = r'C:\Users\A0067477\3DArterySegmentation\nnUNet_raw_data_base\nnUnet_raw_data\Task501_Artery\labelsTR'


def dicom2niigz(patient, path, path_to_nnUnet_raw, seg_exists, path_to_nnUnet_label = ''):
    
    path_to_raw = glob.glob(os.path.join(path, patient, r'*/*/*'))
    
    pat = 'PAT_' + str(patient[-3::]) + '_0000.nii.gz'
    
    if pat in os.listdir(path_to_nnUnet_raw):
        print('Patient already saved.')
        return
    dicom2nifti.dicom_series_to_nifti(path_to_raw[0], os.path.join(path_to_nnUnet_raw, pat), reorient_nifti = False)
    
    if seg_exists:
        path_to_label = os.path.join(path, patient, patient+'SegmentationFinal.nii.gz')
        label = 'PAT_' + str(patient[-3::]) + '.nii.gz'
        
        path_to_nnUnet_label_2 = os.path.join(path_to_nnUnet_label, label)
        
        img_nib = nib.load(os.path.join(path_to_nnUnet_raw, pat))
        
        header = img_nib.header
        
        lab_nib = nib.load(path_to_label)
        
        if len(np.unique(lab_nib.get_fdata())) < 5:
            print('Artery is missing in ' + str(patient))
            
        lab = nib.Nifti1Image(lab_nib.get_fdata(), img_nib.affine.copy())
        nib.save(lab, path_to_nnUnet_label_2)
    print('Saved ' + patient)


'''
for patient in tqdm(os.listdir(path)[8::]):
    print(patient)
    if '_' in patient:
        continue
    path_to_raw = os.path.join(path, patient, walk_through)
    path_to_label = os.path.join(path, patient, patient +'SegmentationFinal.nii.gz')
    
    path_to_raw_glob = glob.glob(path_to_raw)
    
    pat = 'PAT_' + str(patient[-3::]) + '_0000.nii.gz'
    label = 'PAT_' + str(patient[-3::]) +'.nii.gz'
    path_to_nnUnet_label_2 = os.path.join(path_to_nnUnet_label, label)
    
    dicom2nifti.dicom_series_to_nifti(path_to_raw_glob[0], os.path.join(path_to_nnUnet_raw, pat), reorient_nifti = False)
    
    img_nib = nib.load(os.path.join(path_to_nnUnet_raw, pat))
    

    
    header = img_nib.header
    
    lab_nib = nib.load(path_to_label)
    if len(np.unique(lab_nib.get_fdata())) < 5:
        print('Artery is missing in ' +str(patient))
        
    lab = nib.Nifti1Image(lab_nib.get_fdata(), img_nib.affine.copy())
    nib.save(lab, path_to_nnUnet_label_2)

'''   