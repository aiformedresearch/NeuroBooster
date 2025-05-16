
import models.backbones.resnet as resnet
import torch
from torch import nn
from models.VICReg import Projector

class init_medbooster(nn.Module):
    def __init__(self, args):
        super().__init__()

        ### imaging
        backbone, num_nodes_embedding = resnet.__dict__[args.backbone](zero_init_residual=True, num_channels=3)
        self.backbone = backbone
        self.head = Projector(args, num_nodes_embedding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x
