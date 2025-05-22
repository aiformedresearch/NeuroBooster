
import models.backbones.resnet as resnet
from models.backbones.beit_vision_transformer import beit_encoder
from models.backbones import deit_vision_transformer
import torch
from torch import nn
from models.VICReg import Projector
# from models import MAE_finetune_model # this has learnable pos embedding
from models import MAE_pretrain_model_no_masking_no_decoder

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

        self.model = MAE_pretrain_model_no_masking_no_decoder.__dict__[args.mae_model](
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

    