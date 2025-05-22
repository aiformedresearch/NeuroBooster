import torch
import models.backbones.resnet as resnet
from torch import nn
from models import MAE_pretrain_model_no_masking_no_decoder

# the following code is from VICReg implementation https://github.com/facebookresearch/vicreg

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class init_vicreg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_features = int(args.projector.split("-")[-1])

        if 'beit' in args.backbone:
            self.encoder = beit_encoder_for_vicreg(args)
            self.projector = Projector(args, self.encoder.embed_dim)

        else:
            self.encoder, self.embedding = resnet.__dict__[args.backbone](
                zero_init_residual=True
            )
            self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        z_x = self.projector(self.encoder(x))
        z_y = self.projector(self.encoder(y))

        return z_x, z_y

class init_vicreg_deit(nn.Module):
    def __init__(self, args):
        super().__init__()

        ####################### MODEL and optimization
        self.model = MAE_finetune_model.__dict__[args.mae_model](
            num_classes=args.num_classes,
            global_pool=False,
        )

        num_nodes_embedding = self.model.head.in_features
        self.model.head = Projector(args, num_nodes_embedding)
        
    def forward(self, x, y):
        z_x = self.model(x)
        z_y = self.model(y)

        return z_x, z_y

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.projector}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

