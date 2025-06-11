import torchio as tio

class TrainTransform_Crop_Affine_Noise_3D(object):
    def __init__(self, args, train_set_mean=None, train_set_std=None):
        # Compute how much to crop from each spatial dimension (assuming cubic volumes)
        remove_voxels = int((256 - args.resize_shape) / 2)

        self.transform = tio.Compose([
            # Crop evenly on all sides
            tio.Crop(cropping=(remove_voxels, remove_voxels, remove_voxels)),

            # Random affine transformation (scaling, rotation, translation)
            tio.RandomAffine(
                scales=(0.85, 1.15),
                degrees=(-15, 15),
                translation=(-5, 5),
                isotropic=True,
                default_pad_value='minimum',
            ),

            # Random smooth bias field
            tio.RandomBiasField(coefficients=(0, 0.05)),

            # Random gamma correction with log-uniform sampling
            tio.RandomGamma(log_gamma=(-0.1625, 0.1398)),

            # Add random Gaussian noise
            tio.RandomNoise(mean=0, std=(0, 0.1)),
        ])

    def __call__(self, sample):
        # sample must be a torchio.Subject or torchio.Image
        return self.transform(sample)


class TrainTransform_Crop_3D(object):
    def __init__(self, args, train_set_mean=None, train_set_std=None):
        remove_voxels = int((256 - args.resize_shape) / 2)

        self.transform = tio.Compose([
            tio.Crop(cropping=(remove_voxels, remove_voxels, remove_voxels))
        ])

    def __call__(self, sample):
        return self.transform(sample)
