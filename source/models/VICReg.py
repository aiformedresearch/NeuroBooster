import torch
import models.backbones.resnet as resnet
import models.backbones.resnet_3D as resnet_3D
from models.backbones.resnet_3D import generate_model_with_output_dim
from torch import nn
from models.backbones import deit_vision_transformer, deit_vision_transformer_no_masking_no_decoder

# the following code is from VICReg implementation https://github.com/facebookresearch/vicreg

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class init_vicreg_3D(nn.Module):
    def __init__(self, args):
        super().__init__()

        if 'beit' in args.backbone:
            self.backbone = beit_encoder(args)
            self.head = Projector(args, self.backbone.embed_dim)

        elif args.backbone.endswith('_3D'):
            # Parse depth from backbone name
            depth_str = args.backbone.replace('resnet', '').replace('_3D', '')
            model_depth = int(depth_str)

            # Instantiate 3D ResNet
            self.encoder, num_nodes_embedding = generate_model_with_output_dim(
                model_depth,
                n_input_channels=1,  # or 3, depending on your input
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                shortcut_type='B',
                widen_factor=1.0,
                n_classes=1  # dummy classifier
            )
            self.encoder.fc = nn.Identity()  # remove classification head
            self.head = Projector(args, num_nodes_embedding)

        else:
            backbone, num_nodes_embedding = resnet.__dict__[args.backbone](
                zero_init_residual=True,
                num_channels=3
            )
            self.projector = Projector(args, num_nodes_embedding)

    def forward(self, x, y):
        z_x = self.projector(self.encoder(x))
        z_y = self.projector(self.encoder(y))

        return z_x, z_y

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
        self.model = deit_vision_transformer_no_masking_no_decoder.__dict__[args.mae_model](
            num_classes=args.num_classes,
            global_pool=False,
        )

        num_nodes_embedding = self.model.head.in_features
        self.model.head = Projector(args, num_nodes_embedding)

    def forward(self, x, y):
        z_x = self.model(x)
        z_y = self.model(y)
        return z_x, z_y

    @property
    def head(self):
        return self.model.head

    @head.setter
    def head(self, value):
        self.model.head = value


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

