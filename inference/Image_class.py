import os
import torchio as tio
from utils.utils import get_logger
logger = get_logger('Image class')

class Image():
    def __init__(self, pat: str,
                 path_to_raw: str):
        
        self.pat = pat
        self.path_to_raw = path_to_raw
        self.was_resized = False
        
    def pather(self):
        raw_path = os.path.join(self.path_to_raw, self.pat)
        return raw_path
    
    def get_images(self):
        raw_path = self.pather()
        
        subject = tio.Subject({
                'image': tio.ScalarImage(raw_path),
                'patID': self.pat,
                'was_resized': False,
                'shape_before_resizing': tuple()})
        subject = self.preprocess_images(subject)
        subject.was_resized = self.was_resized
        subject.shape_before_resizing = self.original_shape
        return subject
    
    def preprocess_images(self, subject):
        transforms_preprocessing = [
                tio.ToCanonical(),
                tio.Clamp(out_min = -180, out_max = 820), #else it is -180, 820
                tio.CopyAffine('image')]
        transform = tio.Compose(transforms_preprocessing)
        
        subject = transform(subject)
        self.original_shape = subject.shape

        
        if subject.shape[-1] < 500:
            logger.info(f'Subject upsampled due to small size: {subject.shape[-1]}')
            resize = tio.Resize(target_shape = (512,512, 640), image_interpolation= 'bspline') #use slice thickness as parameter instead of 3.
            self.was_resized = True
            subject = resize(subject)
        return subject