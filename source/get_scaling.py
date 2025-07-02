# libraries
import torch
from torch import nn
from torchvision import transforms
from pathlib import Path
import argparse
import time
import json
from sklearn import metrics
from datetime import datetime
import numpy as np
import joblib

# utils
import utils.general_utils as general_utils
import utils.dataset_utils as dataset_utils
from utils.optim_utils import EarlyStopping

# models
from models.backbones import resnet
from models.backbones.beit_vision_transformer import beit_small 
from models.SimMIM import load_pretrained_simim

from models.backbones import deit_vision_transformer
from models.MAE_pretrain_model import interpolate_pos_embed

# models
from models.VICReg import init_vicreg, init_vicreg_deit, Projector
from models.MedBooster import init_medbooster, init_medbooster_deit


def str2bool(v):
    if v.lower() in ('true', '1'):
        return True

    elif v.lower() in ('false', '0'):
        return False

def get_arguments():

    parser = argparse.ArgumentParser(
        description="Fine tune and evaluate a pre-trained model", add_help=False
    )
    parser.add_argument("--paradigm", type=str, default="supervised", help='all the pre-trained models are fine-tuned with a supervised learning, however this argument is used to save correctly the results of the fine-tuning, thus here indicate the pre-training paradigm used among the choices: supervised, medbooster, vicreg, simim') # all the models are fine


    # Data
    parser.add_argument("--dataset_name", type=Path, default="AGE", required=True, help='ADNI or AGE dataset')
    parser.add_argument("--images_dir", type=Path, default="/path/to/dataset", required=True,
                        help='Path to the images data')
    parser.add_argument("--tabular_dir", type=Path, default=None,
                        help='Path to the tabular data')
    parser.add_argument('--train_classes_percentage_values', 
                        nargs='*', 
                        default = None,
                        action = general_utils.keyvalue,
                        help = 'dictionary with the percentage of samples to use for each class in the training set in case it is desired to control the frequency class distribution')
    parser.add_argument("--resize_shape", type = int, default =224, help='Resiz all images')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument('--labels_percentage', 
                        type = int,
                        default = 100)
    parser.add_argument("--balanced_val_set", type = str2bool, default =True, help='True to sample the validation set to get a uniform class frequency distribution, it works only for the classification task')
    parser.add_argument("--normalization", type=str, default='')
    parser.add_argument(
        "--cross_val_folds",
        default=1,
        type=int,
        help="number of folds for cross validation, if 1, it will perform a bootstrap",
    )

    # Model architecture
    parser.add_argument("--mae_model", type=str, default="mae_vit_small_patch16", help='Architecture of MAE')
    parser.add_argument("--backbone", type=str, default="resnet34", help='Architecture of the backbone encoder network, e.g., resnet34, beit_small')
    parser.add_argument("--pretrained_path", type=Path, default = 'exp', help="path to pre-trained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )

    # Optimization
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--fine-tune-batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--val-batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--freeze_backbone",
        default=1,
        type=int,
        choices=(1, 0),
        help="freeze or finetune backbone weights",
    )
    parser.add_argument(
        "--weighted_loss",
        type=int,
        default = 1,
        help="weight loss with inverse class frequency",
    )
    
    # Others
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    parser.add_argument('--device', default='cuda:0',
                    help='device to use for training / testing')


    parser.add_argument("--simim_bottleneck", type = int, default =1, help='simMIM model param: bottleneck')
    parser.add_argument("--simim_depth", type = int, default =12, help='simMIM model param: depth')
    parser.add_argument("--simim_mlp_ratio", type = int, default =4, help='simMIM model param: mlp ratio')
    parser.add_argument("--simim_num_heads", type = int, default =6, help='simMIM model param: number heads')
    parser.add_argument("--simim_emb_dim", type = int, default =384, help='simMIM model param: embedding dim')
    parser.add_argument("--simim_encoder_stride", type = int, default =16, help='simMIM model param: encoder stride')
    parser.add_argument("--simim_in_chans", type = int, default =3, help='simMIM model param: depth')
    parser.add_argument("--simim_use_bn", type = str2bool, default =True, help='simMIM model param: use batch normalization')
    parser.add_argument("--simim_patch_size", type = int, default =16, help='simMIM model param: patch size')
    parser.add_argument("--simim_mask_patch_size", type = int, default =32, help='simMIM data augmentation param: mask patch size')
    parser.add_argument("--simim_mask_ratio", type = float, default =0.5, help='simMIM data augmentation param: mask ratio')
    parser.add_argument("--simim_drop_path_rate", type = float, default =0.1, help='simMIM data augmentation param: drop path rate')

    parser.add_argument('--mae_input_size', default=224, type=int,
                    help='images input size')

    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=20)

    # others
    parser.add_argument("--seed", type=int, default=0)


    return parser

def main(args):
    main_worker(args.device, args)


def main_worker(gpu, args):

    gpu = torch.device(args.device)
    args.gpu = gpu
    args.exp_dir_original = args.exp_dir
    args.num_classes = 1
    args.pre_training_paradigm = args.paradigm
    if (args.backbone != 'beit_small') and (args.pre_training_paradigm == 'simim'):
        print(f'{args.backbone} is being changed to "beit_small" which is the only tested architecture for simim in this repo')
        args.backbone='beit_small'
    args.paradigm = 'supervised' # all the pre-trained models are fine-tuned with a supervised approa

    # prepare folds:
    if 'AGE' in str(args.dataset_name):
        dataset_name = 'AGE'
        args.task = 'regression'
        criterion = nn.MSELoss().cuda(gpu) 

    elif 'ADNI' in str(args.dataset_name):
        dataset_name = 'ADNI'
        args.task = 'classification'

    if args.task == 'classification':

        if args.train_classes_percentage_values is None:
            args.train_classes_percentage_values = {}
        else:
            args.train_classes_percentage_values = {int(key): int(value) for key, value in args.train_classes_percentage_values.items()}

    if args.cross_val_folds == 1:
        indexes_train_folds, indexes_val_folds, targets_train_folds, targets_val_folds, _, tabular_scaler_folds, args.num_classes  = dataset_utils.bootstrap(args)
    else:
        indexes_train_folds, indexes_val_folds, targets_train_folds, targets_val_folds, _, tabular_scaler_folds, args.num_classes  = dataset_utils.cross_validation(args)

    ######## run for each fold: 
    print('for fold')
    for fold in range(args.cross_val_folds):
        print(f'################ fold {fold} ################')
        args.exp_dir = args.exp_dir_original/ f'fold_{fold}'
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        
        ######################## DATASET 
        indexes_train_fold_i = indexes_train_folds[fold]
        targets_train_fold_i = targets_train_folds[fold]
        indexes_val_fold_i = indexes_val_folds[fold]
        targets_val_fold_i = targets_val_folds[fold]
        tabular_scaler_fold_i = tabular_scaler_folds[fold]
        scaler_output_path = args.exp_dir / "tabular_scaler.pkl"
        joblib.dump(tabular_scaler_fold_i, scaler_output_path)
        print(f"Scaler for fold {fold} saved to {scaler_output_path}")
        print(f'TRAINING number of samples after considering labels percentage: {len(targets_train_fold_i)} here reported: {targets_train_fold_i}')
        print(f'VALIDATION number of samples after considering labels percentage: {len(targets_val_fold_i)} here reported: {targets_val_fold_i}')
        criterion_MAE = nn.L1Loss() # just to compute an additional metric to compare results with other works on the same dataset

if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    print('setting reproducibility')
    general_utils.set_reproducibility(args.seed)
    main(args)



