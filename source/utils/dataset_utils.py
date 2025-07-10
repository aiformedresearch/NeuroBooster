import torch
import random
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold 
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from pathlib import Path
from PIL import Image
import os
import data_augmentations.medbooster_augmentations
import data_augmentations.simim_augmentations
import data_augmentations.vicreg_augmentations

from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torch
import random

from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torch
import random

from torch.utils.data import Dataset
import torch
import nibabel as nib
import numpy as np
import random


class ADNI_AGE_Dataset_3D(Dataset):
    """
    Load actual 3D NIfTI volumes (already shaped as 3D data).
    """

    def __init__(self, args, fold_targets, fold_indexes, transforms, train_mean=False, train_std=False):
        self.fold_targets = fold_targets    
        self.fold_indexes = fold_indexes
        self.paradigm = args.paradigm
        self.augmentation_rate = getattr(args, "augmentation_rate", 0)

        # Load NIfTI data
        nifti_path = "/Ironman/scratch/Andrea/data_from_bernadette/AGE_prediction/3D_data/17_01_2024/AgePred_3D.nii.gz"  # replace with your real path
        ramdisk_path = "/dev/shm/AgePred_3D.npy"

        if os.path.exists(ramdisk_path):
            print(f"✅ Loading data from RAM (/dev/shm): {ramdisk_path}")
            self.all_imgs = np.load(ramdisk_path, mmap_mode='r')
        else:
            print(f"🚀 Loading NIfTI and saving to RAM (/dev/shm)")
            nifti_obj = nib.load(nifti_path)
            self.all_imgs = np.asarray(nifti_obj.dataobj, dtype=np.float32)
            np.save(ramdisk_path, self.all_imgs)
            print(f"✅ Saved to {ramdisk_path}")

        #self.all_imgs = nib.load(args.images_dir)
        #self.all_imgs = np.asanyarray(self.all_imgs.dataobj)  # shape: (W, H, D, N)
        self.all_imgs = np.float32(self.all_imgs)

        # Select fold images: (W, H, D, N) → (N, 1, D, H, W)
        self.fold_imgs = self.all_imgs[:, :, :, fold_indexes]             # (W, H, D, selected_N)
        self.fold_imgs = np.transpose(self.fold_imgs, (3, 2, 1, 0))       # (N, D, H, W)
        self.fold_imgs = np.expand_dims(self.fold_imgs, axis=1)          # (N, 1, D, H, W)
        self.fold_imgs = self.fold_imgs / 255.0
        self.fold_imgs = 2.0 * self.fold_imgs - 1.0  # Normalize to [-1, 1]

        # Normalize statistics
        all_pixels = self.fold_imgs.reshape(self.fold_imgs.shape[0], -1)
        if train_mean:
            self.mean = train_mean
            self.std = train_std
            print(f'mean pixel from train set: {self.mean}, std pixels from train set: {self.std}')
        else:
            self.mean = np.mean(all_pixels)
            self.std = np.std(all_pixels)
            print(f'mean pixel: {self.mean}, std pixels: {self.std}')

        # Transforms
        self.transform = transforms[0](args, self.mean, self.std)
        if len(transforms) > 1:
            self.default_transform = transforms[1](args, self.mean, self.std)
        else:
            self.default_transform = self.transform

    def __getitem__(self, index):
        # Get 3D volume: shape [1, D, H, W]
        volume = torch.tensor(self.fold_imgs[index])  # Already [1, D, H, W]

        # Apply transforms
        if self.paradigm in ['vicreg', 'supervised', 'medbooster', 'mae']:
            transform_fn = self.transform if random.random() < self.augmentation_rate else self.default_transform
            img_x = transform_fn(volume)
            img_y = self.transform(volume) if self.paradigm == 'vicreg' else torch.zeros(1)
            tabular = torch.zeros(1) if self.paradigm in ['vicreg', 'mae'] else self.fold_targets[index]
        
        elif self.paradigm in ['medbooster_corrupted']:
        
            #pil_img = transforms.ToPILImage()(original_img)
            if random.random() < self.augmentation_rate:
                img_x =  self.transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            else:
                img_x = self.default_transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            
            img_y = torch.zeros(1)

            tabular = self.fold_targets[index]  
            #print('tabular from loader', tabular)
            tabular = self.corrupt(torch.tensor(tabular, dtype=torch.float), self.marginal_distributions, self.corruption_rate)

        elif self.paradigm == 'simim':
            transform_fn = self.transform if random.random() < self.augmentation_rate else self.default_transform
            img_x, img_y = transform_fn(volume)
            tabular = torch.zeros(1)
        else:
            raise Exception('paradigm not found')

        return img_x, img_y, tabular, volume, self.fold_indexes[index]

    def __len__(self):
        return self.fold_imgs.shape[0]




class ADNI_AGE_Dataset(Dataset):
    """
    Load the dataset.
    """

    def __init__(self, args, fold_targets, fold_indexes, transforms, train_mean=False, train_std=False):
        self.fold_targets = fold_targets    
        self.fold_indexes = fold_indexes

        # Load nifti data
        self.all_imgs = nib.load(args.images_dir)
        self.paradigm = args.paradigm
        self.all_imgs = np.asanyarray(self.all_imgs.dataobj) # numpy array of shape (W,H,C,N)
        self.all_imgs = np.float32(self.all_imgs)

        self.fold_imgs = self.all_imgs[:,:,:,fold_indexes] 

        self.fold_imgs = np.swapaxes( self.fold_imgs, 0,3)
        self.fold_imgs = np.swapaxes(self.fold_imgs, 1,2) # after swapping axes, array shape (N,C,H,W)
        self.fold_imgs = self.fold_imgs/255.0
        self.fold_imgs = 2.0*self.fold_imgs-1.0
        all_pixels = self.fold_imgs.reshape(self.fold_imgs.shape[0], -1)
        if train_mean:
            self.mean = train_mean
            self.std = train_std
            print(f'mean pixel from train set: {self.mean}, std pixels from train set: {self.std}')
        else:
            self.mean =np.mean(all_pixels)
            self.std = np.std(all_pixels)
            print(f'mean pixel: {self.mean}, std pixels: {self.std}')
        self.transform = transforms[0](args, self.mean, self.std)
            
        if len(transforms)>1:
            self.augmentation_rate = args.augmentation_rate
            self.default_transform = transforms[1](args, self.mean, self.std)
        else:
            self.default_transform = transforms[0](args, self.mean, self.std)
            self.augmentation_rate = 0

        if args.paradigm == 'medbooster_corrupted':
            data_df = pd.DataFrame(fold_targets, columns = [f'feature_{n}' for n in range(len(fold_targets[0])) ], dtype = float)
            self.marginal_distributions = data_df.transpose().values.tolist()
            self.corruption_rate = args.corruption_rate
            self.corrupt = transforms[2]

    def __getitem__(self, index):

        original_img = torch.tensor(self.fold_imgs[index,:,:,:]) 

        if self.paradigm in ['vicreg', 'simclr']:
            if random.random() < self.augmentation_rate:
                img_x =  self.transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            else:
                img_x = self.default_transform(original_img)
                img_x = img_x.repeat(3, 1, 1)

            img_y = self.transform(original_img).repeat(3, 1, 1)
            tabular = torch.zeros(1) # no usage of tabular features

        elif self.paradigm in ['supervised', 'medbooster', 'mae']:
            
            if random.random() < self.augmentation_rate:
                img_x =  self.transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            else:
                img_x = self.default_transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            img_y = torch.zeros(1)
            if self.paradigm == 'mae':
                tabular = torch.zeros(1) # no use of tabular features
            else:
                tabular = self.fold_targets[index]

        elif self.paradigm in ['medbooster_corrupted']:
            
            if random.random() < self.augmentation_rate:
                img_x =  self.transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            else:
                img_x = self.default_transform(original_img)
                img_x = img_x.repeat(3, 1, 1)
            img_y = torch.zeros(1)
            tabular = self.fold_targets[index]


        elif self.paradigm in ['simim']:
            if random.random() < self.augmentation_rate:
                img_x, img_y =  self.transform(original_img) # image, mask
            else:
                img_x, img_y =  self.default_transform(original_img) 

            img_x = img_x.repeat(3, 1, 1)
            tabular = torch.zeros(1)
        else:  
            raise Exception('paradigm not found')       
        return img_x, img_y, tabular, original_img, self.fold_indexes[index]

    def __len__(self):
        return self.fold_imgs.shape[0] # image shape: [N x C x H x W]
    
def check_data_leakage(groups, train_indexes, val_indexes):
    # check data leakage again:
    train_groups = list(set([groups[i] for i in train_indexes]))
    val_groups = list(set([groups[i] for i in val_indexes]))
    for subject in train_groups:
        if subject in val_groups:
            raise Exception('data leakage')
        
    for subject in val_groups:
        if subject in train_groups:
            raise Exception('data leakage')
    return

def identity_function(x):
    return x

def cross_validation(args):
    print('cross-validation')
    args.data_info_folder = args.exp_dir/f'data_experiments_info/{args.paradigm}_lab_percent_{args.labels_percentage}'
    targets, groups, num_classes = get_tabular_info(args)
    if args.task == 'classification':
        sets_kfold_splitter =StratifiedGroupKFold(n_splits = args.cross_val_folds, shuffle = False)
    
    elif args.task == 'regression':
        sets_kfold_splitter = GroupKFold(n_splits = args.cross_val_folds)

    indexes = list(range(len(targets)))
    indexes_train_folds = {fold:[] for fold in range(args.cross_val_folds)}
    indexes_val_folds = {fold:[] for fold in range(args.cross_val_folds)}
    targets_train_folds = {fold:[] for fold in range(args.cross_val_folds)}
    targets_val_folds = {fold:[] for fold in range(args.cross_val_folds)}
    tabular_scaler_folds = {fold: identity_function for fold in range(args.cross_val_folds)}

    for fold, (indexes_train_fold_i, indexes_val_fold_i) in enumerate(sets_kfold_splitter.split(indexes, targets, groups)):
        print(f'dataset preparation fold {fold}')
        check_data_leakage(groups, indexes_train_fold_i, indexes_val_fold_i)
            
        targets_train_fold_i = np.array([targets[i] for i in indexes_train_fold_i])
        targets_val_fold_i = np.array([targets[i] for i in indexes_val_fold_i])
        print(f"before scaling train set age mean { np.mean(targets_train_fold_i, axis=0)} , std {np.std(targets_train_fold_i,axis=0)}")
        print(f"before scaling val set age mean {np.mean(targets_val_fold_i, axis=0)} std {np.std(targets_val_fold_i, axis=0)}")
        
        if args.task == 'regression':
            if args.normalization == 'standardization':
                print('standardization')
                tabular_scaler = StandardScaler()
                scale = True
            elif args.normalization == 'minmax':
                print('minmax')
                tabular_scaler = MinMaxScaler(feature_range=(0,1))
                scale = True
            else:
                print('no normalization')
                scale = False
            if scale:
                only_one_target = False
                if len(targets_train_fold_i.shape)==1:
                    only_one_target = True
                    print('only one target')
                    targets_train_fold_i = targets_train_fold_i.reshape(-1, 1) # required for scaler
                    targets_val_fold_i = targets_val_fold_i.reshape(-1,1) # required for scaler
                targets_train_fold_i = tabular_scaler.fit_transform(targets_train_fold_i)
                targets_val_fold_i = tabular_scaler.transform(targets_val_fold_i)
                print(f"after scaling train set age mean { np.mean(targets_train_fold_i, axis=0)} , std {np.std(targets_train_fold_i,axis=0)}")
                print(f"after scaling val set age mean {np.mean(targets_val_fold_i, axis=0)} std {np.std(targets_val_fold_i, axis=0)}")
                tabular_scaler_folds[fold]=tabular_scaler
                
                if only_one_target:
                    targets_train_fold_i = targets_train_fold_i[:,0]
                    targets_val_fold_i = targets_val_fold_i[:,0]
                
        elif args.task == 'classification':
            #tabular_scaler_folds = None
            classes_count_train =get_classes_count([targets[i] for i in indexes_train_fold_i])
            classes_count_val = get_classes_count([targets[i] for i in indexes_val_fold_i])
            min_class_train= min(classes_count_train.values())
            train_cl_0 = [idx for idx in indexes_train_fold_i if targets[idx] == 0]
            train_cl_1 = [idx for idx in indexes_train_fold_i if targets[idx] == 1]
            indexes_train_sampled_fold_i = []
            if args.train_classes_percentage_values[0]>0:
                # arbitrarily balance the dataset classes frequency
                indexes_train_sampled_fold_i_cl0 = random.sample(train_cl_0, max(1,int(min_class_train*args.train_classes_percentage_values[0]/100)))
                print('before usign labels percentage, cl0: ', len(indexes_train_sampled_fold_i_cl0))
                if args.paradigm == 'supervised' or  args.workflow_step == 'fine_tune_evaluate':
                    indexes_train_sampled_fold_i_cl0 = random.sample(indexes_train_sampled_fold_i_cl0, int(len(indexes_train_sampled_fold_i_cl0)*args.labels_percentage/100))
                    print('after usign labels percentage, cl0: ', len(indexes_train_sampled_fold_i_cl0))

                indexes_train_sampled_fold_i_cl1 = random.sample(train_cl_1, max(1,int(min_class_train*args.train_classes_percentage_values[1]/100)))
                print('before usign labels percentage, cl1: ', len(indexes_train_sampled_fold_i_cl1))
                if args.paradigm == 'supervised' or  args.workflow_step == 'fine_tune_evaluate':
                    indexes_train_sampled_fold_i_cl1 = random.sample(indexes_train_sampled_fold_i_cl1, int(len(indexes_train_sampled_fold_i_cl1)*args.labels_percentage/100))
                    print('after usign labels percentage, cl1: ', len(indexes_train_sampled_fold_i_cl1))
                
                indexes_train_sampled_fold_i += indexes_train_sampled_fold_i_cl0
                indexes_train_sampled_fold_i += indexes_train_sampled_fold_i_cl1

            indexes_train_fold_i = indexes_train_sampled_fold_i
            indexes_train_folds[fold] = indexes_train_fold_i

            min_class_val = min(classes_count_val.values())
            val_cl_0 = [idx for idx in indexes_val_fold_i if targets[idx] == 0]
            val_cl_1 = [idx for idx in indexes_val_fold_i if targets[idx] == 1]
            val_index_sampled_fold_i = []
            val_index_sampled_fold_i += random.sample(val_cl_0, min_class_val)
            val_index_sampled_fold_i += random.sample(val_cl_1, min_class_val)
            indexes_val_fold_i = val_index_sampled_fold_i
            
            targets_train_fold_i = np.array([targets[i] for i in indexes_train_fold_i])
            targets_val_fold_i = np.array([targets[i] for i in indexes_val_fold_i])
            
        targets_train_folds[fold] = np.array(targets_train_fold_i) 
        targets_val_folds[fold] = np.array(targets_val_fold_i)
        indexes_train_folds[fold] = np.array(indexes_train_fold_i)
        indexes_val_folds[fold] = np.array(indexes_val_fold_i)

        if args.task == 'classification':
            print(f"after sampling and class percentage imposition, while avoiding data_leakage, classes_count_train: {get_classes_count([targets[i] for i in indexes_train_fold_i])}, classes_count_val: {get_classes_count([targets[i] for i in indexes_val_fold_i])}")

        assert len(targets_train_folds[fold]) == len(indexes_train_fold_i)
        assert len(targets_val_folds[fold]) == len(indexes_val_fold_i)

        check_data_leakage(groups, indexes_train_fold_i, indexes_val_fold_i) # double check
    
    if not(args.paradigm in ['supervised', 'medbooster', 'medbooster_corrupted']):
        targets_train_folds =  {fold:[] for fold in range(args.cross_val_folds)}
        targets_val_folds =  {fold:[] for fold in range(args.cross_val_folds)}

    data_info = {'indexes_train_folds':indexes_train_folds, 'indexes_val_folds': indexes_val_folds, 'targets_train_folds': targets_train_folds, 'targets_val_folds': targets_val_folds , 'all_targets': targets, 'tabular_scaler_folds': tabular_scaler_folds, 'num_classes': num_classes} 
    return data_info.values()

def bootstrap(args):
    print('bootstrap')
    if 'AGE' in str(args.dataset_name):
        args.data_info_folder = args.exp_dir/f'data_experiments_info/{args.paradigm}_lab_percent_{args.labels_percentage}'

        targets, groups, num_classes = get_tabular_info(args)
        indexes = np.array(range(len(targets)))
        indexes_train = random.choices(indexes,k=len(indexes))
        indexes_val = np.setdiff1d(indexes, indexes_train)
        targets_train = targets[indexes_train]
        targets_val = targets[indexes_val]
        check_data_leakage(groups, indexes_train, indexes_val)

        print(f"before scaling train set age mean { np.mean(targets_train, axis=0)} , std {np.std(targets_train,axis=0)}")
        print(f"before scaling val set age mean {np.mean(targets_val, axis=0)}, std {np.std(targets_val, axis=0)}")

        if args.normalization == 'standardization':
            print('standardization')
            tabular_scaler = StandardScaler()
            scale = True
        elif args.normalization == 'minmax':
            print('minmax')
            tabular_scaler = MinMaxScaler(feature_range=(0,1))
            scale = True
        else:
            print('no normalization')
            scale = False
        if scale:
            only_one_target = False
            if len(targets_train.shape)==1:
                only_one_target = True
                targets_train = targets_train.reshape(-1, 1) # required for scaler
                targets_val = targets_val.reshape(-1,1) # required for scaler

            targets_train = tabular_scaler.fit_transform(targets_train)
            #print('after scaling targets:', targets_train_fold_i )
            targets_val = tabular_scaler.transform(targets_val)

            if only_one_target:
                targets_train= targets_train[:,0]
                targets_val = targets_val[:,0]
                    
        if args.paradigm == 'supervised':
            print('before usign labels percentage:')
            indexes_train = indexes_train[:int(len(indexes_train)*args.labels_percentage/100)]
            targets_train = targets_train[:int(len(targets_train)*args.labels_percentage/100)]
            print(f'using {args.labels_percentage}% of the samples, length: {len(indexes_train)}')
        else:
            print('using all the samples because the targets are not the labels for this paradigm pretraining method')
    
    elif 'ADNI' in str(args.dataset_name):
        args.data_info_folder = args.exp_dir/f'data_experiments_info/{args.paradigm}_lab_percent_{args.labels_percentage}'
        targets, groups, num_classes = get_tabular_info(args)
        ########## bootstrap ADNI:
        df = pd.read_csv(args.tabular_dir)
        unique_groups = df['Subject'].unique()
        indexes_for_each_subject = {group:df.index[df['Subject'] == group].tolist()  for group in unique_groups}

        indexes_subjects = np.array(range(len(indexes_for_each_subject.keys())))
        indexes_subjects_train = random.choices(indexes_subjects,k=len(indexes_subjects))
        indexes_subjects_val = np.setdiff1d(indexes_subjects, indexes_subjects_train)

        indexes_train = []
        for index in indexes_subjects_train:
            indexes_train += indexes_for_each_subject[list(indexes_for_each_subject.keys())[index]]

        indexes_val = []
        for index in indexes_subjects_val:
            indexes_val += indexes_for_each_subject[list(indexes_for_each_subject.keys())[index]]

        targets_train = np.array(targets[indexes_train])
        targets_val = np.array(targets[indexes_val])

        ######## make val set balanced:
        classes_count_val = get_classes_count([targets[i] for i in indexes_val])
        min_class_val = min(classes_count_val.values())
        val_cl_0 = [idx for idx in indexes_val if targets[idx] == 0]
        val_cl_1 = [idx for idx in indexes_val if targets[idx] == 1]
        val_index_sampled = []
        val_index_sampled += random.sample(val_cl_0, min_class_val)
        val_index_sampled += random.sample(val_cl_1, min_class_val)
        indexes_val = val_index_sampled
        targets_val= np.array([targets[i] for i in indexes_val])

        shuffle_idxs = list(range(len(targets_train)))
        np.random.shuffle(shuffle_idxs)
        indexes_train = np.array(indexes_train)
        indexes_train = indexes_train[shuffle_idxs]
        targets_train = targets_train[shuffle_idxs]

        if args.paradigm == 'supervised' or args.workflow_step == 'fine_tune_evaluate':
            print(f'using {args.labels_percentage}% of the samples')
            indexes_train = indexes_train[:int(len(indexes_train)*args.labels_percentage/100)]
            targets_train = targets_train[:int(len(targets_train)*args.labels_percentage/100)]
        else:
            print('using all the samples because the targets are not the labels for this paradigm pretraining method')

        tabular_scaler = identity_function

    assert len(targets_train) == len(indexes_train)
    assert len(targets_val) == len(indexes_val)

    check_data_leakage(groups, indexes_train, indexes_val) # double check
    
    if not(args.paradigm in ['supervised', 'medbooster', 'medbooster_corrupted']):
        targets_train = []  
        targets_val = []  
    
    data_info = {'indexes_train_folds': [indexes_train], 'indexes_val_folds': [indexes_val], 'targets_train_folds': [targets_train], 'targets_val_folds': [targets_val] , 'all_targets': [targets], 'tabular_scaler_folds': [tabular_scaler], 'num_classes': num_classes}
    return data_info.values()

def get_tabular_info(args):
    df = pd.read_csv(args.tabular_dir)

    if 'AGE' in str(args.dataset_name):
        if args.paradigm =='supervised':
            print('SINGLE TARGET VARIABLE')
            targets = list(df['Age'])
            num_classes = 1
        else:
            print('MULTIPLE TARGET VARIABLE')
            if 'eTIV' in list(df.columns):
                list_feature_names = [
                    "cortex_FD", "lh_cortex_FD", "rh_cortex_FD", "lh_frontal_cortex_FD",
                    "lh_temporal_cortex_FD", "lh_parietal_cortex_FD", "lh_occipital_cortex_FD",
                    "rh_frontal_cortex_FD", "rh_temporal_cortex_FD", "rh_parietal_cortex_FD",
                    "rh_occipital_cortex_FD", "cortex_CT", "lh_cortex_CT", "rh_cortex_CT",
                    "lh_frontal_cortex_CT", "lh_occipital_cortex_CT", "lh_temporal_cortex_CT",
                    "lh_parietal_cortex_CT", "rh_frontal_cortex_CT", "rh_occipital_cortex_CT",
                    "rh_temporal_cortex_CT", "rh_parietal_cortex_CT", "Left_Caudate",
                    "Left_Putamen", "Left_Pallidum", "Left_Hippocampus", "Left_Amygdala",
                    "Left_Accumbens_area", "Right_Caudate", "Right_Putamen", "Right_Pallidum",
                    "Right_Hippocampus", "Right_Amygdala", "Right_Accumbens_area",
                    "WM_hypointensities", "SubCortGrayVol", "Left_Thalamus", "Right_Thalamus",
                    "lhCorticalWhiteMatterVol", "rhCorticalWhiteMatterVol", "lhCortexVol",
                    "rhCortexVol"
                ]

            else:
                list_feature_names = ['cortex_FD','lh_cortex_FD','rh_cortex_FD','lh_frontal_cortex_FD','lh_temporal_cortex_FD','lh_parietal_cortex_FD','lh_occipital_cortex_FD','rh_frontal_cortex_FD','rh_temporal_cortex_FD','rh_parietal_cortex_FD','rh_occipital_cortex_FD','cortex_CT','lh_cortex_CT','rh_cortex_CT','lh_frontal_cortex_CT','lh_occipital_cortex_CT','lh_temporal_cortex_CT','lh_parietal_cortex_CT','rh_frontal_cortex_CT','rh_occipital_cortex_CT','rh_temporal_cortex_CT','rh_parietal_cortex_CT']
            num_classes = len(list_feature_names)
            df_features = df[list_feature_names]
            targets = df_features.to_numpy() # features as labels

    elif 'ADNI' in str(args.dataset_name):
        print("targets: 0 = Cognitive Normal, 1 = Alzheimer's disease")
        targets = list(df['Group'])
        targets = [0 if item == 'CN' else item for item in targets]
        targets = [1 if item == 'AD' else item for item in targets]
        num_classes = 1

    groups = list(df['Subject'])

    return np.array(targets), np.array(groups), num_classes

def get_classes_count(targets):
    dataset_info = {}
    for cl in set(list(targets)):
        count = list(targets).count(cl)
        dataset_info[cl] = count
        print(f'cl: {cl}, count: {count} ')
    
    return dataset_info

