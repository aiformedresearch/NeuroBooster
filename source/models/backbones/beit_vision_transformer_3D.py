import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

# -----------------------------------------------------------------------------
# This implementation adapts a standard 2D BEiT-style Vision Transformer (ViT)
# to operate on 3D volumetric inputs. The changes are minimal but essential to
# support 3D medical or scientific imaging tasks. Here is what has changed:
#
# - The input shape has changed from (B, C, H, W) to (B, C, D, H, W), reflecting
#   the transition from 2D images to 3D volumes.
#
# - The Patch Embedding layer now uses Conv3d instead of Conv2d. This allows us
#   to extract non-overlapping 3D patches from the volume. The resulting tensor
#   is flattened into a sequence of patch embeddings, just like in the 2D case.
#
# - The number of patches is computed based on the 3D grid defined by the patch
#   size and the input volume dimensions.
#
# - The positional embedding is a learnable parameter with shape
#   (1, num_patches + 1, embed_dim), where the +1 accounts for the [CLS] token.
#   This structure is identical to 2D ViT, except that the number of patches
#   corresponds to 3D volume tokens.
#
# - All other components, including the Transformer encoder blocks, attention,
#   MLP layers, dropout, and layer norms, remain unchanged. These modules operate
#   on sequences of token embeddings, and the change from 2D to 3D is handled
#   entirely within the embedding layer.
#
# - This architecture does not use masking or a decoder. It is specifically
#   designed to be used in VICReg (self-supervised learning) and supervised
#   classification tasks. The model outputs the [CLS] token representation.
#
# This is a typical and effective approach for converting 2D ViTs to 3D. It is
# widely used in the medical imaging field (e.g., TransBTS, ViT-V-Net) and in
# video models like TimeSformer or ViViT, which also operate on 3D data (time + H + W).
# -----------------------------------------------------------------------------

def to_3tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """ 3D Volume to Patch Embedding """
    def __init__(self, img_size=(64, 128, 128), patch_size=(16, 16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = [img_size[i] // patch_size[i] for i in range(3)]
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerEncoder3D(nn.Module):
    """
    Vision Transformer Encoder for 3D volume input.
    Adapted from BEiT/SimMIM for use in VICReg and Supervised Learning.

    Key features:
    - 3D Conv patch embedding
    - No masking or decoder
    - Returns CLS token as representation
    - Can be used for both VICReg (self-supervised) and classification (supervised)
    """
    def __init__(self, img_size=(64, 128, 128), patch_size=(16, 16, 16), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token only


def beit_encoder_3d(args):
    return VisionTransformerEncoder3D(
        img_size=args.volume_shape,  # (D, H, W)
        patch_size=args.simim_patch_size,  # e.g., (16, 16, 16)
        in_chans=args.simim_in_chans,
        embed_dim=args.simim_emb_dim,
        depth=args.simim_depth,
        num_heads=args.simim_num_heads,
        mlp_ratio=args.simim_mlp_ratio,
        drop_path_rate=args.simim_drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
