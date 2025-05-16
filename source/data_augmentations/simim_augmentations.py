import torchvision.transforms as T
from torchvision.transforms import transforms, InterpolationMode
import numpy as np
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

class TrainTransform_CropMask:
    def __init__(self, args, train_mean, train_std):
        remove_pixels_w = remove_pixels_h = int((256-args.resize_shape)/2)
        self.transform_img = T.Compose([
                UnsqueezeTransform(dim=-1),
                torchio.transforms.Crop(cropping = (remove_pixels_w, remove_pixels_h,0)), 
                SqueezeTransform(),
        ])
        self.mask_generator = MaskGenerator(
            input_size=args.resize_shape,
            mask_patch_size=args.simim_mask_patch_size,
            model_patch_size=args.simim_patch_size,
            mask_ratio=args.simim_mask_ratio,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask
    
class TrainRandomResizeRotateMask:
    def __init__(self, args, train_mean, train_std):
        remove_pixels_w = remove_pixels_h = int((256-args.resize_shape)/2)
        self.transform_img = T.Compose([
                UnsqueezeTransform(dim=-1),
                torchio.transforms.Crop(cropping = (remove_pixels_w, remove_pixels_h,0)), 
                torchio.transforms.RandomAffine(scales = (0.85,1.8), degrees = (-15,15), translation = (-15,15,-15,15,0,0), isotropic = True, default_pad_value = 'minimum'),
                torchio.transforms.RandomBiasField((-0.05, 0.05)),
                torchio.transforms.RandomGamma(log_gamma = (-0.1625, 0.1398)),#(-0.1625, 0.1398)), # gamma 0.85,1.15 -> ln(0.85) = -0.1625, ln(1.15) = 0.1398
                torchio.transforms.RandomNoise(mean = 0, std = (0,0.1)),
                SqueezeTransform(),
        ])

        self.mask_generator = MaskGenerator(
            input_size=args.resize_shape,
            mask_patch_size=args.simim_mask_patch_size,
            model_patch_size=args.simim_patch_size,
            mask_ratio=args.simim_mask_ratio,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask

# The following code is from SimMIM implementation https://github.com/microsoft/SimMIM
# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
