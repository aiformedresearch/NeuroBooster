import torch
import torch.nn as nn
import torch
import models.backbones.resnet as resnet
import models.backbones.resnet_3D as resnet_3D
from models.backbones.resnet_3D import generate_model_with_output_dim
from torch import nn
from models.backbones import deit_vision_transformer, deit_vision_transformer_no_masking_no_decoder
from models.VICReg import Projector, Projector_3D

class init_simclr_deit(nn.Module):
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

class init_simclr(nn.Module):
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

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: two augmented versions of the same input batch
        Returns:
            z1, z2: projected embeddings (used in SimCLR loss)
        """
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        return z1, z2
