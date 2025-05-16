
# The following code is from SimMIM implementation https://github.com/microsoft/SimMIM

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from collections import Counter
from bisect import bisect_right
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
import json
from functools import partial
from torch import optim as optim


TRAIN_EPOCHS = 300
TRAIN_WARMUP_EPOCHS = 20
TRAIN_WEIGHT_DECAY = 0.05
TRAIN_BASE_LR = 5e-4
TRAIN_WARMUP_LR = 5e-7
TRAIN_MIN_LR = 5e-6
# Clip gradient norm
TRAIN_CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
TRAIN_AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
TRAIN_ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
TRAIN_USE_CHECKPOINT = False

TRAIN_LR_SCHEDULER_NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
TRAIN_LR_SCHEDULER_DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
TRAIN_LR_SCHEDULER_DECAY_RATE = 0.1
# Gamma / Multi steps value, used in MultiStepLRScheduler
TRAIN_LR_SCHEDULER_GAMMA = 0.1
TRAIN_LR_SCHEDULER_MULTISTEPS = []

# Optimizer
TRAIN_OPTIMIZER_NAME = 'adamw'
# Optimizer Epsilon
TRAIN_OPTIMIZER_EPS = 1e-8
# Optimizer Betas
TRAIN_OPTIMIZER_BETAS = (0.9, 0.999)
# SGD momentum
TRAIN_OPTIMIZER_MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
TRAIN_LAYER_DECAY = 1.0

def build_optimizer(config, model, is_pretrain):
    if is_pretrain:
        return build_pretrain_optimizer(config, model)
    else:
        return build_finetune_optimizer(config, model)


def build_pretrain_optimizer(config, model):
    #logger.info('>>>>>>>>>> Build Optimizer for Pre-training Stage')
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        #logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        #logger.info(f'No weight decay keywords: {skip_keywords}')

    parameters = get_pretrain_param_groups(model,  skip, skip_keywords)

    # opt_lower = args.optim.lower()
    # optimizer = None
    # if opt_lower == 'sgd':
    #     optimizer = optim.SGD(parameters, momentum=TRAIN_OPTIMIZER_MOMENTUM, nesterov=True,
    #                           lr=TRAIN_BASE_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    #elif opt_lower == 'adamw':
    optimizer = optim.AdamW(parameters, eps=TRAIN_OPTIMIZER_EPS, betas=TRAIN_OPTIMIZER_BETAS,
                                lr=TRAIN_BASE_LR, weight_decay=TRAIN_WEIGHT_DECAY)

    #logger.info(optimizer)
    return optimizer
    

def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    #logger.info(f'No decay params: {no_decay_name}')
    #logger.info(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def build_finetune_optimizer(config, model):
    #logger.info('>>>>>>>>>> Build Optimizer for Fine-tuning Stage')

    num_layers = 12 #config.MODEL.VIT.DEPTH
    get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)

    scales = list(TRAIN_LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
    
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        #logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        #logger.info(f'No weight decay keywords: {skip_keywords}')

    parameters = get_finetune_param_groups(
        model, TRAIN_BASE_LR, TRAIN_WEIGHT_DECAY,
        get_layer_func, scales, skip, skip_keywords)
    
    opt_lower = TRAIN_OPTIMIZER_NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=TRAIN_OPTIMIZER_MOMENTUM, nesterov=True,
                              lr=TRAIN_BASE_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=TRAIN_OPTIMIZER_EPS, betas=TRAIN_OPTIMIZER_BETAS,
                                lr=TRAIN_BASE_LR, weight_decay=TRAIN_WEIGHT_DECAY)

    #logger.info(optimizer)
    return optimizer


def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # logger.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(TRAIN_EPOCHS * n_iter_per_epoch)
    warmup_steps = int(TRAIN_WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(TRAIN_LR_SCHEDULER_DECAY_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in TRAIN_LR_SCHEDULER_MULTISTEPS]

    lr_scheduler = None
    if TRAIN_LR_SCHEDULER_NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            #t_mul=1.,
            lr_min=TRAIN_MIN_LR,
            warmup_lr_init=TRAIN_WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif TRAIN_LR_SCHEDULER_NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=TRAIN_WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif TRAIN_LR_SCHEDULER_NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=TRAIN_LR_SCHEDULER-DECAY_RATE,
            warmup_lr_init=TRAIN_WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif TRAIN_LR_SCHEDULER_NAME == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=TRAIN_LR_SCHEDULER_GAMMA,
            warmup_lr_init=TRAIN_WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        
        assert self.warmup_t <= min(self.milestones)
    
    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None