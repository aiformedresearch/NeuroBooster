import torch
import torch.nn as nn

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
