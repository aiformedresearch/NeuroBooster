import torch.nn.functional as F

# The following code is from SimMIM implementation https://github.com/microsoft/SimMIM

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

def simim_loss(x, x_rec, mask, in_chans, patch_size):
    # image reconstruction loss
    mask = mask.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1).contiguous()
    loss_recon = F.l1_loss(x, x_rec, reduction='none')
    loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) /in_chans
    return loss