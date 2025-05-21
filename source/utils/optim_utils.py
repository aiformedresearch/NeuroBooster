import math
import torch
from torch import optim

class EarlyStopping:
    def __init__(self, patience=5, min_epochs=200):
        """
        Args:
            patience (int): Number of epochs with no significant training loss change after which training will be stopped.
            verbose (bool): If True, prints a message when training is stopped due to early stopping.
            delta (float): Minimum change in training loss to qualify as a significant change.
        """
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.min_epochs = min_epochs
        self.best_loss = 1e8

    def __call__(self, train_loss, epoch):
        if epoch < self.min_epochs:
            if train_loss < self.best_loss:
                self.best_loss = train_loss

        if epoch > self.min_epochs:

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.counter = 0
            else:
                print(f'loss not getting better: best loss {self.best_loss}, current loss: {train_loss}, counter: {self.counter}')
                self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True

        


## the following code is from VICReg implementation https://github.com/facebookresearch/vicreg

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def exclude_bias_and_norm(p):
    return p.ndim == 1


def adjust_learning_rate_mae(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.mae_warmup_epochs:
        lr = args.mae_lr * epoch / args.mae_warmup_epochs 
    else:
        lr = args.mae_min_lr + (args.mae_lr - args.mae_min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.mae_warmup_epochs) / (args.epochs - args.mae_warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1