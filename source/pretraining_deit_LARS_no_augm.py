# libraries
import torch
from torch import nn
from pathlib import Path
import argparse
import time
import json
import copy
from datetime import datetime

# utils
import utils.general_utils as general_utils
import utils.dataset_utils as dataset_utils
from utils.optim_utils import LARS, adjust_learning_rate, exclude_bias_and_norm, EarlyStopping, adjust_learning_rate_mae
import timm.optim.optim_factory as optim_factory
import math
from utils.model_utils import NativeScalerWithGradNormCount as NativeScaler

# models
from models.VICReg import init_vicreg, init_vicreg_deit
from models.MedBooster import init_medbooster, init_medbooster_deit
from models.SimMIM import init_simim
from models import MAE_pretrain_model

# data augmentations
import data_augmentations
from data_augmentations import simim_augmentations, mae_augmentations

# losses
from losses.VICReg_loss import vicreg_loss
from losses.SimMIM_loss import simim_loss
from losses.MedBooster_loss import medbooster_loss


def str2bool(v):
    if v.lower() in ('true', '1'):
        return True

    elif v.lower() in ('false', '0'):
        return False

def get_arguments():

    # PARADIGM for pre-training
    parser = argparse.ArgumentParser(description="Pre-train a model with a paradigm of choice", add_help=False)
    parser.add_argument("--paradigm", type=str, default="supervised", help='choices: supervised medbooster vicreg simim mae')

    # Data directory
    parser.add_argument("--dataset_name", type=Path, default="AGE", required=True, help='ADNI or AGE dataset')
    parser.add_argument("--images_dir", type=Path, default="/path/to/dataset", required=True, help='Path to the images data')
    parser.add_argument("--tabular_dir", type=Path, default=None, help='Path to the tabular data')
    parser.add_argument("--cross_val_folds", default=0, type=int,help="number of folds for cross validation")
    parser.add_argument("--resize_shape", type = int, default = 224, help='Resize all images to this shape')
    parser.add_argument('--labels_percentage', type = int, default = 100, help="If supervised paradigm, select the percentage of labels to be used")
    parser.add_argument("--normalization", type=str, default='')
    parser.add_argument("--augmentation_rate", type = float, default = 0.9, help='paradigm data augmentation hyperparameter')

    # Results directory
    parser.add_argument("--exp-dir", type=Path, default="./exp", help='Path to the experiment folder, where all the outputs will be stored')

    # Model architecture
    parser.add_argument("--backbone", type=str, default="resnet34", help='Architecture of the backbone encoder network, e.g., resnet34, beit_small')
    parser.add_argument("--projector", default="1024-1024", help='Projector layers number of nodes')

    # Paradigm specific params:
    parser.add_argument("--vicreg_sim_coeff", type=float, default=25.0, help='vicreg, Invariance regularization loss coefficient')
    parser.add_argument("--vicreg_std_coeff", type=float, default=25.0, help='vicreg, Variance regularization loss coefficient')
    parser.add_argument("--vicreg_cov_coeff", type=float, default=1.0, help='vicreg, Covariance regularization loss coefficient')
    
    parser.add_argument("--simim_bottleneck", type = int, default =1, help='SimMIM model param: bottleneck')
    parser.add_argument("--simim_depth", type = int, default =12, help='SimMIM model param: depth')
    parser.add_argument("--simim_mlp_ratio", type = int, default =4, help='SimMIM model param: mlp ratio')
    parser.add_argument("--simim_num_heads", type = int, default =6, help='SimMIM model param: number heads')
    parser.add_argument("--simim_emb_dim", type = int, default =384, help='SimMIM model param: embedding dim')
    parser.add_argument("--simim_encoder_stride", type = int, default =16, help='SimMIM model param: encoder stride')
    parser.add_argument("--simim_in_chans", type = int, default =3, help='SimMIM model param: depth')
    parser.add_argument("--simim_use_bn", type = str2bool, default =True, help='SimMIM model param: use batch normalization')
    parser.add_argument("--simim_patch_size", type = int, default =16, help='SimMIM model param: patch size')
    parser.add_argument("--simim_mask_patch_size", type = int, default =32, help='SimMIM data augmentation param: mask patch size')
    parser.add_argument("--simim_mask_ratio", type = float, default =0.5, help='SimMIM data augmentation param: mask ratio')
    parser.add_argument("--simim_drop_path_rate", type = float, default =0.1, help='SimMIM data augmentation param: drop path rate')

    parser.add_argument('--mae_model', default='mae_vit_small_patch16', type=str,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mae_mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mae_norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(mae_norm_pix_loss=True)
    parser.add_argument('--mae_weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--mae_lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--mae_blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--mae_min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--mae_warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')
    parser.set_defaults(mae_pin_mem=True)
    parser.add_argument('--mae_world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--mae_local_rank', default=-1, type=int)
    parser.add_argument('--mae_dist_on_itp', action='store_true')
    parser.add_argument('--mae_dist_url', default='env://',
                        help='url used to set up distributed training')

    # Optimization
    parser.add_argument("--epochs", type=int, default=100, help='Maximum number of epochs')
    parser.add_argument("--min_epochs", type=int, default=250, help='Minimum number of epochs')
    parser.add_argument("--batch-size", type=int, default=128, help='batch size')
    parser.add_argument("--base_lr", type=float, default=0.2, help='learning rate')
    parser.add_argument("--optim", type=str, default='SGD', help='optimizer')
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--weighted_loss",
        type=str2bool,
        help="weight loss with inverse class frequency",
    )

    # Others
    parser.add_argument('--device', default='cuda:0', help='device to be used for training the model')
    parser.add_argument('--comment', default='', type=str, help='leave a comment about the experiment')
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    return parser

def main(args):
    
    print(f'paradigm: {args.paradigm}')
    ######### setup:
    gpu = torch.device(args.device)
    args.gpu = gpu
    args.exp_dir_original = args.exp_dir
    args.workflow_step = 'pretraining' # needed because we call some functions both in the script for pretraining and finetuning

    # type of task:
    if 'AGE' in str(args.dataset_name): 
        args.task = 'regression'
    elif 'ADNI' in str(args.dataset_name): 
        args.task = 'classification'
    else:
        print('dataset not recognized, check the data path or create your own dataset function for processing it')
    
    if args.cross_val_folds == 1:
        # bootstrapping
        indexes_train_folds, _, targets_train_folds, _, _, tabular_scaler_folds, args.num_classes  = dataset_utils.bootstrap(args)
    else: 
        # cross validation
        indexes_train_folds, _, targets_train_folds, _, _, tabular_scaler_folds, args.num_classes  = dataset_utils.cross_validation(args)
    
    ######## run for each fold:
    for fold in range(args.cross_val_folds):
        print(f'############# fold {fold} ##############')
        last_state = {}

        args.exp_dir = args.exp_dir_original/ f'fold_{fold}'
        args.exp_dir.mkdir(parents=True, exist_ok=True)

        # prepare files and folders to save info in real time during the execution of the script
        if (args.backbone != 'beit_small') and (args.paradigm == 'simim'):
            print(f'{args.backbone} is being changed to "beit_small" which is the only tested architecture for simim in this repo')
            args.backbone='beit_small'
        all_stats_file = open(args.exp_dir / "pretraining_all_stats.txt", "a", buffering=1)
        model_info_file = open(args.exp_dir / "pretraining_model_info.txt", "a", buffering=1)
        save_df_train_path = plot_df_train_path = args.exp_dir / 'pretraining_metrics'        

        # prepare data for the fold
        indexes_train_fold_i = indexes_train_folds[fold]
        targets_train_fold_i = targets_train_folds[fold]
        tabular_scaler_fold_i = tabular_scaler_folds[fold]

        ################### DATA AUGMENTATION
        if args.paradigm == 'vicreg':
            transforms = [data_augmentations.vicreg_augmentations.TrainTransform_Crop_Affine_Noise, # augmentation
                          data_augmentations.vicreg_augmentations.TrainTransform_Crop, # default transform
                          ]

        elif (args.paradigm == 'supervised') or (args.paradigm == 'medbooster'):
            transforms = [data_augmentations.medbooster_augmentations_no_augm.TrainTransform_Crop_Affine_Noise, # augmentation
                          data_augmentations.medbooster_augmentations_no_augm.TrainTransform_Crop, # default transform
                          ]

        elif (args.paradigm == 'mae'):
            #transforms = [data_augmentations.neuro_booster_augmentations.TrainTransform_Resize_Norm]
            transforms = [mae_augmentations.TrainTransform_Crop_Affine_Noise, # augmentation
                          mae_augmentations.TrainTransform_Crop, # default transform
                          ]

        elif args.paradigm == 'simim':
            transforms = [data_augmentations.simim_augmentations.TrainRandomResizeRotateMask,
                          data_augmentations.simim_augmentations.TrainTransform_CropMask]
        else:
            print('paradigm not implemented')

        train_dataset = dataset_utils.ADNI_AGE_Dataset(args, targets_train_fold_i, indexes_train_fold_i, transforms)

        ################## MODEL ARCHITECTURE:
        print('INITIALIZING MODEL')
        if args.paradigm in ['medbooster', 'supervised']:
            projector_dims = args.projector.split("-")
            args.projector = f'{projector_dims[0]}-{args.num_classes}'
        
        print(f'PROJECTOR DIMENSIONS: {args.projector}')

        # if ('beit' in args.backbone) and not(args.paradigm == 'simim'):
        #     raise Exception('the transformer backbone is supposed to be used with simim paradigm')

        ########## LOSS FUNCTION
        if args.paradigm == 'supervised' or args.paradigm == 'medbooster':

            if 'deit' in args.backbone:
                print('init deit backbone')
                model = init_medbooster_deit(args).cuda(gpu)
                print(model)
            else:
                model = init_medbooster(args).cuda(gpu)

            model.head = nn.Sequential(model.head)
            
            if args.task == 'regression':
                print('regression loss')
                criterion = medbooster_loss().cuda(gpu)
            elif args.task == 'classification':
                print('classification loss')
                sampled_classes = [0,1]
                if args.weighted_loss and args.paradigm == 'supervised':
                    class_0_count = len(targets_train_fold_i[targets_train_fold_i==0])
                    class_1_count = len(targets_train_fold_i[targets_train_fold_i==1])
                    print('class_0_count train', class_0_count)
                    print('class_1_count train', class_1_count)
                    pos_weight =torch.tensor(class_0_count/class_1_count)
                    print('using weighted loss')
                else:
                    print('not using weighted loss')
                    pos_weight = None
                print(f'positive class weight: {pos_weight}')
                criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight).cuda(gpu) # sigmoid + BCELoss = BCEWithLogitsLoss
    
        elif args.paradigm =='vicreg':
            if 'deit' in args.backbone:
                model = init_vicreg_deit(args).cuda(gpu)
                print(model)
            else:
                model = init_vicreg(args).cuda(gpu)
            criterion = vicreg_loss(args).cuda(gpu)

        elif args.paradigm == 'mae':
            model = MAE_pretrain_model.__dict__[args.mae_model](norm_pix_loss=args.mae_norm_pix_loss)

        elif args.paradigm == 'simim':
            model = init_simim(args).cuda(gpu)
            criterion = simim_loss

        # if 'deit' in args.backbone:
        #     # ====== BEGIN MAE-DERIVED CODE ======
        #     # Adapted from https://github.com/facebookresearch/mae
        #     # Licensed under CC BY-NC 4.0
        #     print(args.mae_lr)
        #     if args.mae_lr is None:  # only base_lr is specified
        #         args.accum_iter = 1
        #         eff_batch_size = args.batch_size * args.accum_iter * 1 #misc.get_world_size() =1
        #         args.mae_lr = args.mae_blr * eff_batch_size / 256
        #     param_groups = optim_factory.add_weight_decay(model, args.mae_weight_decay)
        #     optimizer = torch.optim.AdamW(param_groups, lr=args.mae_lr, betas=(0.9, 0.95))
        #     print(optimizer)
        #     loss_scaler = NativeScaler()
        #     # ====== END MAE-DERIVED CODE ======

        # else:
        param_groups = model.parameters()

        ######### OPTIMIZATION:
        #### loader setup
        drop_last=False
        if len(train_dataset)>args.batch_size:
            print('dropping last batches')
            drop_last = True
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            worker_init_fn=general_utils.seed_worker,
            generator=g,
            drop_last=drop_last,
        )

        # # optimizer
        # if ('beit' in args.backbone) or ('deit' in args.backbone):
        #     lr = 5e-4

        if True:
            if args.optim == 'SGD':
                print('optimizer: SGD')
                optimizer = torch.optim.SGD(param_groups, lr=0)
            
            elif args.optim == 'LARS':
                print('optimizer: LARS')
                optimizer = LARS(
                param_groups,
                lr=0,
                weight_decay=args.weight_decay,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm,
                )

            elif args.optim == 'ADAM':
                print('optimizer: adamw')
                optimizer = torch.optim.AdamW(
                param_groups,
                lr=0,
                weight_decay=args.weight_decay,
                )

        if False:
            from utils.simim_optim import build_scheduler,build_optimizer 
            optimizer = build_optimizer(args, model, is_pretrain=True)
            lr_scheduler = build_scheduler(args, optimizer, len(loader))
            num_steps = len(loader)

        # SAVE MODEL ARCHITECTURE AND OPTIMIZER:
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size(), file = model_info_file)

        # Print optimizer's state_dict
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name], file = model_info_file)

        num_params, total_size_MB = general_utils.compute_model_size(model)
        print(f"Number of parameters: {num_params}", file = model_info_file)
        print(f"Total size (MB): {total_size_MB:.2f} MB", file = model_info_file)

        ############ START PRE-TRAINING
        start_epoch = 0
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler() # to deal with underflow of gradients when using autocast
        df_train_metrics = {}
        max_gpu_usage = 0
        model.requires_grad_(True)
        model = model.cuda(gpu)
        best_loss = 1e10
        accum_iter = 1                
        
        if fold == 0:
            arg_dict = vars(args)
            with open(args.exp_dir/'args_pretraining.txt', "w") as file:
                for key, value in arg_dict.items():
                    if value is not None:
                        file.write(f"{key}: {value}\n")

        early_stopping = EarlyStopping(patience=args.patience, min_epochs = args.min_epochs)

        for epoch in range(start_epoch, args.epochs):
            starting_epoch = True

            # if 'deit' in args.backbone:
            #     optimizer.zero_grad()

            for step, (img_x, img_y, tabular, original_img, samples_id) in enumerate(loader, start = epoch * len(loader)):   
                img_x = img_x.to(torch.float32)  
                img_y = img_y.to(torch.float32) 

                #if not('deit' in args.backbone):
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    if args.paradigm == 'mae':
                        loss, _, _ = model(img_x.cuda(gpu,non_blocking=True), mask_ratio=args.mae_mask_ratio)
                        step_metrics = {'loss':loss.item()}
                    
                    elif args.paradigm == 'simim':
                        img_x = img_x.cuda(gpu,non_blocking=True)
                        mask = img_y.cuda(gpu,non_blocking=True)
                        img_rec, mask = model(img_x, mask)
                        loss = criterion(img_x, img_rec, mask, model.in_chans, model.patch_size)
                        step_metrics = {'loss':loss.item()}
                    
                    elif args.paradigm == 'vicreg':
                        z_x,z_y = model.forward(img_x.cuda(gpu, non_blocking=True), img_y.cuda(gpu, non_blocking=True))
                        loss, sim_loss, std_loss, cov_loss = criterion(z_x,z_y)
                        step_metrics = {'loss': loss.item(), 'sim_loss': sim_loss.item(), 'std_loss':std_loss.item(), 'cov_loss':cov_loss.item()}

                    elif (args.paradigm == 'medbooster') or (args.paradigm == 'supervised') :
                        output = model.forward(img_x.cuda(gpu,non_blocking=True))
                        tabular = tabular.to(torch.float16)
                        if len(tabular.shape)==1:
                            tabular = tabular.reshape(-1,1)
                        loss = criterion(output, tabular.cuda(gpu, non_blocking=True))
                        with torch.no_grad():
                            if args.task == 'regression':
                                if tabular_scaler_fold_i:
                                    rescaled_loss = criterion(torch.tensor(tabular_scaler_fold_i.inverse_transform(output.cpu())).cuda(gpu,non_blocking=True), torch.tensor(tabular_scaler_fold_i.inverse_transform(tabular.cpu())).cuda(gpu,non_blocking=True))
                                else:
                                    rescaled_loss = loss
                                
                                step_metrics = {'loss':loss.item(), 'rescaled_loss':rescaled_loss.item()}
                            else:
                                step_metrics = {'loss':loss.item()}
                    
                if True: #not(args.backbone in ['deit','beit']) and (args.paradigm in ['supervised', 'medbooster','vicreg']):
                    lr = adjust_learning_rate(args, optimizer, loader, step)

                # if 'deit' in args.backbone:
                #     # ====== BEGIN MAE-DERIVED CODE ======
                #     # Adapted from https://github.com/facebookresearch/mae
                #     # Licensed under CC BY-NC 4.0

                #     # per iteration (instead of per epoch) lr scheduler
                #     adjust_learning_rate_mae(optimizer, step / len(loader) + epoch, args)
                    
                #     #samples = samples.to(device, non_blocking=True)
                #     loss_value = loss.item()

                #     if not math.isfinite(loss_value):
                #         print("Loss is {}, stopping training".format(loss_value))
                #         sys.exit(1)

                #     lr = optimizer.param_groups[0]["lr"]
                #     step_metrics = {'loss':loss.item()}
                #     # ====== END MAE-DERIVED CODE ======

                #     loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
                #     optimizer.zero_grad()
                #     # ====== END MAE-DERIVED CODE ======
                
                # else:
                scaler.scale(loss).backward() # to avoid underflow of gradients when using autocast
                # if ('deit' in args.backbone) or ('beit' in args.backbone):
                #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                #     if False:
                #         lr_scheduler.step_update(epoch * num_steps + step)
                scaler.step(optimizer)
                scaler.update()

                # compute avg loss:
                with torch.no_grad():
                    if starting_epoch: 
                        starting_epoch = False
                        train_all_metrics_dict = step_metrics   
                    else: 
                        for name_loss_i, loss_i in step_metrics.items():
                            train_all_metrics_dict[name_loss_i] += loss_i/len(loader)
            
            if starting_epoch:
                print('not enough samples:', len(loader))
                break

            ############## PLOT AND SAVE METRICS AT THE END OF EACH EPOCH
            with torch.no_grad():
                all_stats = dict(
                epoch=epoch,
                time=int(time.time() - start_time),
                base_lr = args.base_lr,
                lr=lr,
                day_time = str(datetime.now()),
                )    

                for metric_name, metric in train_all_metrics_dict.items():
                    all_stats[metric_name] = metric

                print(all_stats)
                print(json.dumps(all_stats), file=all_stats_file)        
                df_train_metrics = general_utils.update_df_metrics(df_train_metrics, epoch, train_all_metrics_dict, save_df_train_path, plot_df_train_path, 'train' )
                    
                if loss < best_loss:
                    best_loss = loss
                    best_state = create_dict_state(args, model, optimizer, epoch)
                
                # Check for early stopping
                early_stopping(train_all_metrics_dict['loss'], epoch = epoch)
            
                if early_stopping.early_stop:
                    last_state = create_dict_state(args, model, optimizer, epoch)
                    torch.save(last_state, args.exp_dir / f"last_pretrained.pth")
                    torch.save(best_state, args.exp_dir / f"best_pretrained.pth")
                    print(f"EARLY STOPPAGE, epoch {epoch}")
                    break
    
        print('No early stoppage, saving model to path:', args.exp_dir / f"pretrained.pth")
        last_state = create_dict_state(args, model, optimizer, epoch)
        torch.save(last_state, args.exp_dir / f"last_pretrained.pth")
        torch.save(best_state, args.exp_dir / f"best_pretrained.pth")
        (args.exp_dir / "pretraining_done.txt").touch()


def create_dict_state(args, model, optimizer, epoch):        
    if (args.paradigm in ['simim']) or ('beit' in args.backbone) :
        state = dict(
        epoch=epoch + 1,
        model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        )
    elif (args.paradigm in ['supervised','medbooster']) and ('resnet' in args.backbone):
        state = dict(
        epoch=epoch + 1,
        backbone=copy.deepcopy(model.backbone.state_dict()),
        head=copy.deepcopy(model.head.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        )

    elif (args.paradigm in ['vicreg']) and ('resnet' in args.backbone):
        state = dict(
        epoch=epoch + 1,
        backbone=copy.deepcopy(model.encoder.state_dict()),
        head=copy.deepcopy(model.projector.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        )           

    elif 'deit' in args.backbone:
        state = dict(
        epoch=epoch + 1,
        model=copy.deepcopy(model.state_dict()),
        optimizer=optimizer.state_dict(),
        )   

    return state      

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Pre-training script', parents=[get_arguments()])
    args = parser.parse_args()
    general_utils.set_reproducibility(args.seed)
    main(args)
    torch.cuda.empty_cache()







