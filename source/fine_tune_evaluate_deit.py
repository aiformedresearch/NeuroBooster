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
    args.paradigm = 'supervised' # all the pre-trained models are fine-tuned with a supervised approa
    sigmoid = nn.Sigmoid()
    args.workflow_step = 'fine_tune_evaluate'


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
    for fold in range(args.cross_val_folds):
        print(f'################ fold {fold} ################')
        args.exp_dir = args.exp_dir_original/ f'fold_{fold}'
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        
        #### finetuning on ADNI based on the pretraining done with AGE dataset:
        # the logic is that the pretraining is done with the AGE dataset, then the model then can be fine-tuned on the AGE or ADNI dataset to evaluate the out of distribution performance
        # For the fine-tuning on the ADNI dataset: the labels percentage of this script controls the percentage of samples to use for the fine-tuning, however the pre-trained model is the one that assumed 100% of labels was available for the pre-training with the supervised learning
        if (not((args.pre_training_paradigm == 'supervised')) or dataset_name == 'ADNI') and (args.labels_percentage < 100):
            args.pretrained_path = Path(str(args.exp_dir).replace(dataset_name,'AGE').replace(f'labels_percentage_{args.labels_percentage}','labels_percentage_100'))/'best_pretrained.pth'
        else:
            args.pretrained_path = Path(str(args.exp_dir).replace(dataset_name,'AGE'))/'best_pretrained.pth'

        print(f'pretrained model from: {args.pretrained_path}')
        all_stats_file = open(args.exp_dir / "fine_tune_and_evaluate_all_stats.txt", "a", buffering=1)
        model_info_file = open(args.exp_dir / "fine_tune_and_evaluate_model_info.txt", "a", buffering=1)
        save_df_train_path = plot_df_train_path = args.exp_dir / 'finetuning_metrics'
        save_df_val_path = plot_df_val_path = args.exp_dir / 'val_metrics'

        ######################## DATASET 
        indexes_train_fold_i = indexes_train_folds[fold]
        targets_train_fold_i = targets_train_folds[fold]
        indexes_val_fold_i = indexes_val_folds[fold]
        targets_val_fold_i = targets_val_folds[fold]
        tabular_scaler_fold_i = tabular_scaler_folds[fold]
        print(f'TRAINING number of samples after considering labels percentage: {len(targets_train_fold_i)} here reported: {targets_train_fold_i}')
        print(f'VALIDATION number of samples after considering labels percentage: {len(targets_val_fold_i)} here reported: {targets_val_fold_i}')

        if args.task == 'classification':
            sampled_classes = [0,1]
            if (args.train_classes_percentage_values[sampled_classes[0]] >0 and args.train_classes_percentage_values[sampled_classes[1]]>0 and args.weighted_loss and args.paradigm == 'supervised'):
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
            monitor_criterion = nn.BCEWithLogitsLoss(pos_weight = None).cuda(gpu) 
        
        #### load dataset Data loading code
        train_transform = [TrainTransform_Resize] 
        val_transform = [ValTransform_Resize]
        train_dataset = dataset_utils.ADNI_AGE_Dataset(args, targets_train_fold_i, indexes_train_fold_i, train_transform)
        val_dataset = dataset_utils.ADNI_AGE_Dataset(args, targets_val_fold_i, indexes_val_fold_i, val_transform, train_mean=train_dataset.mean, train_std=train_dataset.std)

        ####################### MODEL and optimization
        if 'mae' in args.pre_training_paradigm:
            # ====== BEGIN MAE-DERIVED CODE ======
            # Adapted from https://github.com/facebookresearch/mae
            # Licensed under CC BY-NC 4.0

            model = deit_vision_transformer.__dict__[args.mae_model](
                num_classes=args.num_classes,
                global_pool=False,
            )
            num_nodes_embedding = model.head.in_features #192 if tiny
            print('num_nodes_embedding', num_nodes_embedding)
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.pretrained_path)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f'loaded pretrained with msg:', msg)

            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.head.named_parameters():
                p.requires_grad = True

            # ====== END MAE-DERIVED CODE ======
        elif 'beit' in args.backbone:
            backbone = beit_small(args)
            num_nodes_embedding = args.simim_emb_dim #192 if 'tiny' in args.backbone else 384 if 'small' in args.backbone else 768 if 'base' in args.backbone else 1024 if 'large' in args.backbone else 1280
            msg=load_pretrained_simim(args, backbone)
            print(f'loaded pretrained with msg: {msg}')
            backbone.head = None

        elif 'resnet' in args.backbone: 
            backbone, num_nodes_embedding = resnet.__dict__[args.backbone](zero_init_residual=True, num_channels=3)
            state_dict = torch.load(args.pretrained_path, map_location='cpu')   
            msg = backbone.load_state_dict(state_dict["backbone"], strict=True)
            print(f'loaded pretrained with msg: {msg}')
        else:
            print(f'{args.backbone} is not among the possible backbones')
            break

        if args.num_classes == 2:
            num_nodes = 1
        else:
            num_nodes = args.num_classes
        head = nn.Linear(num_nodes_embedding, num_nodes)
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
        head.requires_grad_(True)

        param_groups = [dict(params=head.parameters(), lr=args.lr_head)]

        if args.pre_training_paradigm == 'mae':
            model.head = head 
            print("Model = %s" % str(model))
            n_parameters_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('number of params (M): %.2f' % (n_parameters_model / 1.e6))
            for param_tensor in model.state_dict():
                print('model including head', param_tensor, "\t", model.state_dict()[param_tensor].size(), file = model_info_file)

        else:
            if args.freeze_backbone:
                print('backbone freezed')
                backbone.requires_grad_(False)

            else:
                print('backbone not freezed')
                backbone.requires_grad_(True)
                param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))


            for param_tensor in backbone.state_dict():
                print('backbone', param_tensor, "\t", backbone.state_dict()[param_tensor].size(), file = model_info_file)
            
            for param_tensor in head.state_dict():
                print('head', param_tensor, "\t", head.state_dict()[param_tensor].size(), file = model_info_file)

        
        optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay) #, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        # Print optimizer's state_dict
        for var_name in optimizer.state_dict():
            print('optimizer', var_name, "\t", optimizer.state_dict()[var_name], file = model_info_file)

        ######################### TRAIN AND EVALUATE:
        g = torch.Generator()
        g.manual_seed(args.seed)

        ### loader setup
        drop_last=False
        if len(train_dataset)>args.fine_tune_batch_size:
            print('dropping last batches')
            drop_last = True
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.fine_tune_batch_size, num_workers=args.num_workers,
            pin_memory=True, sampler=None, shuffle = True, worker_init_fn=general_utils.seed_worker, generator=g, drop_last= drop_last
            )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers,
            pin_memory=True, shuffle = False, worker_init_fn=general_utils.seed_worker, generator=g,
            )
        
        start_time = time.time()
        start_epoch = 0
        scaler = torch.cuda.amp.GradScaler()
        
        df_train_metrics = {}
        df_val_metrics = {}
        outputs_and_targets_validation_path = args.exp_dir / 'valdiation_outputs_and_targets'
        outputs_and_targets_validation_path.mkdir(parents=True, exist_ok=True)
        if args.pre_training_paradigm == 'mae':
            model.to(args.gpu)
        else:
            backbone.to(args.gpu)
            head.to(args.gpu)

        early_stopping = EarlyStopping(patience=args.patience, min_epochs = args.min_epochs)
        criterion_MAE = nn.L1Loss() # just to compute an additional metric to compare results with other works on the same dataset
        for epoch in range(start_epoch, args.epochs):
            starting_epoch = True

            if args.pre_training_paradigm == 'mae':
                model.train(True)
                accum_iter = 1
                optimizer.zero_grad()
            else:
                ############# FINETUNING:
                head.train()
                if args.freeze_backbone:
                    backbone.eval() # to set the mode, so that layers like batchn ormalization behave properly
                else:
                    backbone.train() # to set the mode, so that layers like batchn ormalization behave properly

            for step, (img_x, img_y, tabular, original_img, samples_id) in enumerate(train_loader, start=epoch * len(train_loader)):
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():  # automaitc mixed precision
                    if args.pre_training_paradigm == 'mae':
                        output = model(img_x.cuda(gpu, non_blocking=True))

                    elif 'beit' in args.backbone:
                        if args.freeze_backbone:
                            with torch.no_grad():
                                if epoch == 0 and step == 0 and fold == 0: print('backbone freezed')
                                backbone.requires_grad_(False)
                                output = backbone.forward_blocks(img_x.cuda(gpu, non_blocking=True))
                        else:
                            backbone.requires_grad_(True)
                            output = backbone.forward_blocks(img_x.cuda(gpu, non_blocking=True))
                        output = head(output)

                    elif 'resnet' in args.backbone:
                        if args.freeze_backbone:
                            with torch.no_grad():
                                if epoch == 0 and step == 0 and fold == 0: print('backbone freezed')
                                backbone.requires_grad_(False)
                                output = backbone(img_x.cuda(gpu, non_blocking=True))
                        else:
                            backbone.requires_grad_(True)
                            output = backbone(img_x.cuda(gpu, non_blocking=True))
                        output = head(output)

                    tabular = tabular.to(torch.float16)
                    if len(tabular.shape)==1:
                        tabular = tabular.reshape(-1,1)
                    finetuning_loss = criterion(output, tabular.cuda(gpu, non_blocking=True))

                    with torch.no_grad():
                            tabular_scaler_fold_i = tabular_scaler_folds[fold]
                            if args.task == 'regression':
                                if tabular_scaler_fold_i:
                                    finetuning_rescaled_loss = criterion(torch.tensor(tabular_scaler_fold_i.inverse_transform(output.cpu())), torch.tensor(tabular_scaler_fold_i.inverse_transform(tabular.cpu())))
                                else:
                                    finetuning_rescaled_loss = finetuning_loss
                                train_step_metrics = {'fine_tuning_loss': finetuning_loss.item(), 'fine_tuning_rescaled_loss': finetuning_rescaled_loss.item()}
                            else:
                                finetuning_loss_not_weighted = monitor_criterion(output, tabular.cuda(gpu, non_blocking=True))
                                train_step_metrics = {'fine_tuning_loss': finetuning_loss.item(), 'finetuning_loss_not_weighted': finetuning_loss_not_weighted.item()}
                                
                with torch.cuda.amp.autocast():
                    scaler.scale(finetuning_loss).backward() # to avoid underflow of gradients when using autocast
                    scheduler.step()
                    scaler.step(optimizer)
                    scaler.update()

                # compute avg loss:
                with torch.no_grad():
                    if starting_epoch: 
                        starting_epoch = False
                        train_all_metrics_dict = train_step_metrics 
                    else: 
                        for name_loss_i, loss_i in train_step_metrics.items():
                            train_all_metrics_dict[name_loss_i] += loss_i/len(train_loader)
                   
            ########### VALIDATION:
            if args.pre_training_paradigm == 'mae':
                model.eval()
            else:
                backbone.eval()
                head.eval()
            with torch.no_grad():
                y_true = []
                y_pred = []
                starting_epoch = True
                for step, (img_x, img_y, tabular, original_img, samples_id)in enumerate(val_loader, start=epoch * len(val_loader)):
                    with torch.cuda.amp.autocast():  # automaitc mixed precision
                        if args.pre_training_paradigm == 'mae':
                            # img_x = img_x.to(torch.float32)  
                            # img_y = img_y.to(torch.float32) 
                            output = model(img_x.cuda(gpu, non_blocking=True))
                        elif ('deit' in args.backbone) or ('beit' in args.backbone):
                            output = backbone.forward_blocks(img_x.cuda(gpu, non_blocking=True))
                            output = head(output)
                        elif 'resnet' in args.backbone:
                            output = backbone(img_x.cuda(gpu, non_blocking=True))
                            output = head(output)
                        
                        tabular = tabular.to(torch.float16)
                        if len(tabular.shape)==1:
                            tabular = tabular.reshape(-1,1)

                        val_loss = criterion(output, tabular.cuda(gpu, non_blocking=True))

                        with torch.no_grad():
                            if args.task =='regression':
                                if tabular_scaler_fold_i:
                                    val_rescaled_loss = criterion(torch.tensor(tabular_scaler_fold_i.inverse_transform(output.cpu())), torch.tensor(tabular_scaler_fold_i.inverse_transform(tabular.cpu())))
                                    val_MAE_loss = criterion_MAE(torch.tensor(tabular_scaler_fold_i.inverse_transform(output.cpu())), torch.tensor(tabular_scaler_fold_i.inverse_transform(tabular.cpu())))
                                else:
                                    val_rescaled_loss = val_loss
                                    val_MAE_loss = criterion_MAE(output.cpu(), tabular.cpu())
                                
                                val_step_metrics = {'val_loss': val_loss.item(), 'val_rescaled_loss': val_rescaled_loss.item(), 'val_MAE_loss': val_MAE_loss.item() }
                                y_true.extend(list(tabular))
                                y_pred.extend(output.cpu().numpy())
                            elif args.task == 'classification':  
                                y_true.extend(list(tabular))
                                output = sigmoid(output)
                                y_pred.extend(output.cpu().numpy())
                            
                    # compute avg loss:
                    if args.task == 'regression':
                        with torch.no_grad():
                            if starting_epoch: 
                                starting_epoch = False
                                val_all_metrics_dict = val_step_metrics 
                            else: 
                                for name_loss_i, loss_i in val_step_metrics.items():
                                    val_all_metrics_dict[name_loss_i] += loss_i/len(val_loader)
                    else: 
                        val_all_metrics_dict = {}

                ##### compute classification metrics:
                with torch.no_grad():
                    np.save(outputs_and_targets_validation_path / f'y_target_validation_epoch{epoch}.npy', y_true)
                    np.save(outputs_and_targets_validation_path / f'y_predicted_validation_epoch{epoch}.npy', y_pred)

                    if args.task == 'classification': 
                        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(y_pred), pos_label=1)
                        auc = metrics.auc(fpr, tpr)
                        val_all_metrics_dict['auc'] = auc
                        pr_auc = metrics.average_precision_score(np.array(y_true), np.array(y_pred))
                        val_all_metrics_dict['pr_auc'] = pr_auc
                        
            with torch.no_grad():
                all_stats = dict(
                    epoch=epoch,
                    time=int(time.time() - start_time),
                    lr_backbone=args.lr_backbone,
                    lr_head=args.lr_head,
                    day_time = str(datetime.now()),
                )

                for metric_name, metric in train_all_metrics_dict.items():
                    all_stats[metric_name] = metric

                for metric_name, metric in val_all_metrics_dict.items():
                    all_stats[metric_name] = metric

                print(json.dumps(all_stats), file=all_stats_file)
                df_train_metrics = general_utils.update_df_metrics(df_train_metrics, epoch, train_all_metrics_dict, save_df_train_path, plot_df_train_path, 'finetuning' )
                df_val_metrics = general_utils.update_df_metrics(df_val_metrics, epoch, val_all_metrics_dict, save_df_val_path, plot_df_val_path, 'val' )
                            
                # Check for early stopping
                early_stopping(train_all_metrics_dict['fine_tuning_loss'], epoch = epoch)
                if early_stopping.early_stop:
                    print(f"EARLY STOPPAGE, epoch {epoch}")
                    break


# FINE-TUNING DATA AUGMENTATION:
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
import torchio

class TrainTransform_Resize(object):
    def __init__(self, args, train_set_mean, train_set_std):
        remove_pixels_w = remove_pixels_h = int((256-args.resize_shape)/2)
        self.transform = transforms.Compose(
            [   
                UnsqueezeTransform(dim=-1),
                torchio.transforms.Crop(cropping = (remove_pixels_w, remove_pixels_h,0)),
                torchio.transforms.RandomAffine(scales = (1,1), degrees = (-10,10), translation = (-5,5,-5,5,0,0), isotropic = True, default_pad_value = 'minimum'),
                SqueezeTransform(),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        return x1

# VALIDATION DATA AUGMENTATION:
class ValTransform_Resize(object):
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

if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    print('setting reproducibility')
    general_utils.set_reproducibility(args.seed)
    main(args)



