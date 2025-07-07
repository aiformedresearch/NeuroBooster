
import models.backbones.resnet as resnet
import models.backbones.resnet_3D as resnet_3D
from models.backbones.resnet_3D import generate_model_with_output_dim
from models.backbones.beit_vision_transformer import beit_encoder
from models.backbones import deit_vision_transformer, deit_vision_transformer_no_masking_no_decoder
import torch
from torch import nn
from models.VICReg import Projector, Projector_3D
# from models import MAE_finetune_model # this has learnable pos embedding

class init_medbooster_3D(nn.Module):
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
            self.backbone, num_nodes_embedding = generate_model_with_output_dim(
                model_depth,
                n_input_channels=1,  # or 3, depending on your input
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                shortcut_type='B',
                widen_factor=1.0,
                n_classes=1  # dummy classifier
            )
            self.backbone.fc = nn.Identity()  # remove classification head
            self.head = Projector_3D(args, num_nodes_embedding)

        else:
            backbone, num_nodes_embedding = resnet.__dict__[args.backbone](
                zero_init_residual=True,
                num_channels=3
            )
            self.backbone = backbone
            self.head = Projector(args, num_nodes_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class init_medbooster(nn.Module):
    def __init__(self, args):
        super().__init__()

        ### imaging
        if ('beit' in args.backbone):
            self.backbone = beit_encoder(args)
            self.head = Projector(args, self.backbone.embed_dim)

        else:
            backbone, num_nodes_embedding = resnet.__dict__[args.backbone](zero_init_residual=True, num_channels=3)
            self.backbone = backbone
            self.head = Projector(args, num_nodes_embedding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

class init_medbooster_deit(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = deit_vision_transformer_no_masking_no_decoder.__dict__[args.mae_model](
            num_classes=args.num_classes,
            global_pool=False,
        )

        num_nodes_embedding = self.model.head.in_features
        self.model.head = Projector(args, num_nodes_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def head(self):
        return self.model.head

    @head.setter
    def head(self, value):
        self.model.head = value

    