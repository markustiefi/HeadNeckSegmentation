import numpy as np
import torch
import os
import torchio as tio
from model.model import UNet3D
from torch.nn.functional import avg_pool3d

from utils.utils import get_logger
from utils.utils_imageclass import threshold_img, upsample
from inference.Image_class import Image

logger = get_logger('Downsample')
class Downsample(Image):
    def __init__(self, pat: str,
                 path_to_raw: str,
                 gauss_mus: list, gauss_sigma: int,
                 device: str,
                 shape: tuple):
        
        super().__init__(pat, path_to_raw)
        self.gauss_mus = gauss_mus
        self.gauss_sigma = gauss_sigma
        self.shape = shape
        self.device = device
    
    @staticmethod
    def get_sak_size(shape, xy, z):
        # Calculate stride and kernel size for average pooling downsampling.
        stride = (int(shape[-3]/xy), int(shape[-2]/xy), int(shape[-1]/z))
        kernel_size = (int(shape[-3]/xy), int(shape[-2]/xy), int(shape[-1]/z))
        return kernel_size, stride
    
    @staticmethod
    def get_gauss_image(image, mu, sigma):
        # Each pixel is multiplied with a corresponding gaussian value.
        return image*np.exp((-(image-mu)**2)/(2*sigma**2))
    
    def get_gauss_versions(self, img, kernel_size, stride):
        #Solve better later.
        clamp_downsampled = tio.Clamp(out_min = -180, out_max = 820)
        img = clamp_downsampled(img)
        
        # Downsampling using average pooling (smoothed).
        img_down = avg_pool3d(img, kernel_size = kernel_size, stride = stride).float()
        # Normalizing the image.
        img_down = (img_down-img_down.min())/(img_down.max()-img_down.min())
        img_gauss_versions = [img_down]
        
        #assert (len(self.gauss_mus) == len(self.gauss_sigma)) or (type(self.gauss_sigma) is int)
        # Get gaussian versions of the volumes
        for i, mu in enumerate(self.gauss_mus):
            if type(self.gauss_sigma) is int:
                sigma = self.gauss_sigma
            else:
                sigma = self.gauss_sigma[i]
                
            gauss_version = self.get_gauss_image(img, mu = mu, sigma = sigma)
            img_downsampled = avg_pool3d(gauss_version, kernel_size = kernel_size, stride = stride).float()
            img_downsampled = (img_downsampled-img_downsampled.min())/(img_downsampled.max()-img_downsampled.min())
            img_gauss_versions.append(img_downsampled)
            del gauss_version
        return torch.cat(img_gauss_versions)
    
    def downsample(self, subject = None):
        if not subject:
            subject = self.get_images()
        
        img = subject.image.data.long()
        
        xy = self.shape[0]
        z = self.shape[2]

        kernel_size, stride = self.get_sak_size(img.shape, xy, z)
        
        img_downsampled = self.get_gauss_versions(img, kernel_size = kernel_size,
                                                  stride = stride
                                                  )
        del img
        
        subject = tio.Subject(
            image = tio.ScalarImage(tensor = img_downsampled),
            patID = self.pat,
            original_shape = subject.image.shape,
            original_spacing = subject.image.spacing,
            shape_before_CropOrPad = tuple(img_downsampled.shape[1:]),
            original_stride = stride,
            kernel_size = kernel_size,
            gauss_mus = self.gauss_mus,
            was_resized = self.was_resized,
            shape_before_resizing = self.original_shape)
        CropOrPad = tio.CropOrPad(self.shape)
        subject = CropOrPad(subject)
        return subject
    
    def pred_downsampled(self, nets, subject = None, in_channels = 4, out_channels = 4, 
                         f_maps = 32, num_levels = 4, threshold = 0.1, final_sigmoid = False,
                         thr = False, device = 'cuda:0', return_subjectdown = False):
        
        if not subject:
            subject = self.get_images()
            
        subject_down = self.downsample(subject)
        shape_before_CropOrPad = subject_down.shape_before_CropOrPad
        
        model = UNet3D(in_channels = in_channels, out_channels = out_channels, f_maps = f_maps,
                       num_levles = num_levels, final_sigmoid = final_sigmoid)
        prediction = None
        if in_channels == 1:
            raw = subject_down.image.data[0]
            raw = raw.unsqueeze(0).unsqueeze(0)
        elif in_channels == 2:
            raw = subject_down.image.data[[0,2]]
            raw = raw.unsqueeze(0)
        elif in_channels == 3:
            raw = subject_down.image.data[[0,1,3]]
            raw = raw.unsqueeze(0)
        else:
            raw = subject_down.image.data
            raw = raw.unsqueeze(0)
        raw = raw.to(self.device)
        
        for net in nets:
            net = os.path.join(net, 'best_checkpoint.pytorch')

            checkpoint = torch.load(net, map_location = torch.device(self.device))
                
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
                

            with torch.no_grad():
                if prediction is None:
                    prediction = model(raw)
                else:
                    prediction += model(raw)
        
        prediction *= 1/len(nets)
        if not self.device == 'cpu':
            prediction = prediction.cpu()
        # We crop or pad the volume to get consistent shape during training.
        # Invert the operation before we upsample the global prediction mask to the original shape.
        inverse_CropOrPad = tio.CropOrPad(shape_before_CropOrPad)
        prediction = inverse_CropOrPad(prediction.squeeze()).unsqueeze(0)
        print(prediction.shape)
        for k in range(1,2):
            prediction[:,k,:,:,0:40] = 0
        
        transforms_preprocessing = [
                tio.RescaleIntensity(out_min_max = (0,1)),
                tio.CopyAffine('image')]
        transform = tio.Compose(transforms_preprocessing)
        subject = transform(subject)
        
        shape = list(subject_down.original_shape)

        shape = tuple(shape)
        prediction_upsampled = upsample(prediction, shape)
        subject.add_image(image = tio.ScalarImage(tensor = prediction.squeeze().half()), image_name = 'prediction_downsampled')
        subject.add_image(image = tio.ScalarImage(tensor = prediction_upsampled.squeeze().half()), image_name = 'prediction_upsampled')
    
        return subject