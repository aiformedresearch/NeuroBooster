

import torchvision.transforms as transforms
import random
import torch
import torchio

class RandomGammaCorrection(object):
    def __init__(self, start,stop, step):
        self.gamma_corrections = list(range(start,stop, step))

    def __call__(self, img):
        gamma = random.choice(self.gamma_corrections)
        return transforms.functional.adjust_gamma(img, gamma)   

class UnsqueezeTransform(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return torch.unsqueeze(tensor, self.dim)
    
class SqueezeTransform(object):
    def __init__(self,):
        pass

    def __call__(self, tensor):
        return torch.squeeze(tensor)
 
class TrainTransform_Crop_Affine_Noise(object):
    def __init__(self, args, train_set_mean, train_set_std):
        remove_pixels_w = remove_pixels_h = int((256-args.resize_shape)/2)
        self.transform = transforms.Compose(
            [   
                UnsqueezeTransform(dim=-1),
                torchio.transforms.Crop(cropping = (remove_pixels_w, remove_pixels_h,0)),
                torchio.transforms.RandomAffine(scales = (0.85,1.15), degrees = (-15,15), translation = (-5,5,-5,5,0,0), isotropic = True, default_pad_value = 'minimum'),
                torchio.transforms.RandomBiasField((-0.05, 0.05)),
                torchio.transforms.RandomGamma(log_gamma = (-0.1625, 0.1398)), #(-0.1625, 0.1398)), # gamma 0.85,1.15 -> ln(0.85) = -0.1625, ln(1.15) = 0.1398
                torchio.transforms.RandomNoise(mean = 0, std = (0,0.1)),
                SqueezeTransform(),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        return x1

class TrainTransform_Crop(object):
    def __init__(self, args, train_set_mean, train_set_std):
        remove_pixels_w = remove_pixels_h = int((256-args.resize_shape)/2)
        self.transform = transforms.Compose(
            [   
                UnsqueezeTransform(dim=-1),
                torchio.transforms.Crop(cropping = (remove_pixels_w, remove_pixels_h,0)),
                SqueezeTransform(),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        return x1