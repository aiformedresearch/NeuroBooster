import torch.nn.functional as F
from torch import nn
import torch

# the following code is from VICReg implementation https://github.com/facebookresearch/vicreg

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class vicreg_loss(torch.nn.Module):
    def __init__(self, args):
        super(vicreg_loss, self).__init__()
        self.batch_size = args.batch_size
        self.sim_coeff = args.vicreg_sim_coeff
        self.std_coeff = args.vicreg_std_coeff
        self.cov_coeff = args.vicreg_cov_coeff
        self.num_features = int(args.projector.split("-")[-1])    
    
    def forward(self, x, y):
        sim_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)

        cov_y = (y.T @ y) / (self.batch_size - 1)


        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss, sim_loss, std_loss, cov_loss,



