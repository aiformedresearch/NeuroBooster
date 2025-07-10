import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import numpy as np
import argparse
from utils import general_utils, dataset_utils
from models.backbones import deit_vision_transformer
from models.MAE_pretrain_model import interpolate_pos_embed
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchio
from models.backbones.beit_vision_transformer import beit_small 

def str2bool(v):
    return v.lower() in ('true', '1')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=Path, required=True)
    parser.add_argument('--tabular_dir', type=Path, required=True)
    parser.add_argument('--exp_dir', type=Path, required=True)
    parser.add_argument('--resize_shape', type=int, default=224)
    parser.add_argument('--dataset_name', type=str, default='AGE')
    parser.add_argument('--paradigm', type=str, default='supervised')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backbone', default='mae_vit_small_patch16')
    parser.add_argument('--labels_percentage', type=int, default=100)
    parser.add_argument('--cross_val_folds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--normalization', type=str, default='standardization')
    parser.add_argument("--simim_bottleneck", type = int, default =1, help='SimMIM model param: bottleneck')
    parser.add_argument("--simim_depth", type = int, default =12, help='SimMIM model param: depth')
    parser.add_argument("--simim_mlp_ratio", type = int, default =4, help='SimMIM model param: mlp ratio')
    parser.add_argument("--simim_num_heads", type = int, default =6, help='SimMIM model param: number heads')
    parser.add_argument("--simim_emb_dim", type = int, default =384, help='SimMIM model param: embedding dim')
    parser.add_argument("--simim_encoder_stride", type = int, default =16, help='SimMIM model param: encoder stride')
    parser.add_argument("--simim_in_chans", type = int, default =3, help='SimMIM model param: depth')
    parser.add_argument("--simim_use_bn", type = str2bool, default =True, help='SimMIM model param: use batch normalization')
    parser.add_argument("--simim_patch_size", type = int, default =16, help='SimMIM model param: patch size')
    parser.add_argument("--simim_mask_patch_size", type = int, default =32, help='SimMIM data augmentation param: mask patch size')
    parser.add_argument("--simim_mask_ratio", type = float, default =0.5, help='SimMIM data augmentation param: mask ratio')
    parser.add_argument("--simim_drop_path_rate", type = float, default =0.1, help='SimMIM data augmentation param: drop path rate')

    return parser

def generate_saliency_map(model, image, target_value, device):
    image = image.unsqueeze(0).to(device).requires_grad_()
    target = torch.tensor([[target_value]], dtype=torch.float32, device=device)
    output = model(image)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    saliency = image.grad.abs().squeeze().cpu().numpy()
    return saliency.max(axis=0)

def main(args):
    device = torch.device(args.device)
    general_utils.set_reproducibility(args.seed)

    if 'simim' in str(args.exp_dir):
        args.backbone='beit_small'

    args.pretrained_path = args.exp_dir / 'best_finetuned.pth'
    args.task = 'regression'
    args.num_classes = 1

    print(f"Loading backbone: {args.backbone}")

    if 'beit' in args.backbone:
        print(f"Loading BEiT model and head from {args.pretrained_path}")
        backbone = beit_small(args)
        backbone.head = None
        backbone.to(device).eval()

        head = nn.Linear(args.simim_emb_dim, 1).to(device).eval()
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        backbone.load_state_dict(checkpoint['backbone'], strict=True)
        head.load_state_dict(checkpoint['head'], strict=True)

        class BeitWithHead(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
            def forward(self, x):
                x = self.backbone.forward_blocks(x)
                return self.head(x)

        model = BeitWithHead(backbone, head).to(device)
    else:
        model = deit_vision_transformer.__dict__[args.backbone](
            num_classes=args.num_classes,
            global_pool=False
        ).to(device)
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()

    print("Loading validation data and tabular scaler...")
    _, indexes_val_folds, _, targets_val_folds, _, tabular_scaler_folds, _ = dataset_utils.bootstrap(args)
    val_indexes = indexes_val_folds[0]
    val_targets = targets_val_folds[0]
    scaler = tabular_scaler_folds[0]

    transform = [ValTransform_Resize]
    val_dataset = dataset_utils.ADNI_AGE_Dataset(args, val_targets, val_indexes, transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    output_dir = args.exp_dir / 'saliency_maps_average'
    os.makedirs(output_dir, exist_ok=True)

    avg_image_path = output_dir / 'avg_image.npy'
    avg_saliency_path = output_dir / 'avg_saliency.npy'

    if avg_image_path.exists() and avg_saliency_path.exists():
        print(f"✅ Average image and saliency already exist at {output_dir}. Skipping computation.")
        return

    print(f"Loading backbone: {args.backbone}")


    saliency_sum = None
    image_sum = None
    count = 0

    print("Generating and averaging saliency maps...")
    for idx, (img, _, target, _, sample_id) in enumerate(tqdm(val_loader)):
        img = img.to(torch.float32)
        target = target.view(-1, 1).to(torch.float32)

        pred_np = model(img.to(device)).detach().cpu().numpy()
        target_np = target.cpu().numpy()

        if scaler:
            pred_inv = scaler.inverse_transform(pred_np)[0][0]
            target_inv = scaler.inverse_transform(target_np)[0][0]
        else:
            pred_inv = pred_np[0][0]
            target_inv = target_np[0][0]

        saliency = generate_saliency_map(model, img[0], target[0][0].item(), device)
        image_np = img[0].cpu().numpy()
        if image_np.shape[0] == 3:
            image_np = np.mean(image_np, axis=0)
        else:
            image_np = np.squeeze(image_np)

        if saliency_sum is None:
            saliency_sum = saliency.copy()
            image_sum = image_np.copy()
        else:
            saliency_sum += saliency
            image_sum += image_np

        count += 1

    avg_saliency = saliency_sum / count
    avg_image = image_sum / count

    if avg_image.ndim == 3 and avg_image.shape[0] == 3:
        avg_image = np.mean(avg_image, axis=0)

    avg_image = np.rot90(avg_image, 2)
    avg_saliency = np.rot90(avg_saliency, 2)

    # Save arrays
    np.save(output_dir / 'avg_image.npy', avg_image)
    np.save(output_dir / 'avg_saliency.npy', avg_saliency)

    # Save each plot separately
    plt.imshow(avg_image, cmap='gray')
    plt.axis('off')
    plt.title('Average MRI')
    plt.savefig(output_dir / 'average_mri.png', dpi=300)
    plt.close()

    plt.imshow(avg_saliency, cmap='hot')
    plt.axis('off')
    plt.title('Average Saliency Map')
    plt.savefig(output_dir / 'average_saliency.png', dpi=300)
    plt.close()

    plt.imshow(avg_image, cmap='gray')
    plt.imshow(avg_saliency, cmap='hot', alpha=0.5)
    plt.axis('off')
    plt.title('Overlay')
    plt.savefig(output_dir / 'average_overlay.png', dpi=300)
    plt.close()

    # Combined figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(avg_image, cmap='gray')
    axs[0].set_title('Average MRI')
    axs[1].imshow(avg_saliency, cmap='hot')
    axs[1].set_title('Average Saliency Map')
    axs[2].imshow(avg_image, cmap='gray')
    axs[2].imshow(avg_saliency, cmap='hot', alpha=0.5)
    axs[2].set_title('Overlay')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'average_saliency_overlay.png', dpi=300)
    plt.close()

    print(f"✅ Saved average saliency maps, overlays, and arrays to {output_dir}")


class UnsqueezeTransform(object):
    def __init__(self, dim):
        self.dim = dim
    def __call__(self, tensor):
        return torch.unsqueeze(tensor, self.dim)

class SqueezeTransform(object):
    def __call__(self, tensor):
        return torch.squeeze(tensor)

class ValTransform_Resize(object):
    def __init__(self, args, train_set_mean=None, train_set_std=None):
        remove_pixels_w = remove_pixels_h = int((256 - args.resize_shape) / 2)
        self.transform = transforms.Compose([
            UnsqueezeTransform(dim=-1),
            torchio.transforms.Crop(cropping=(remove_pixels_w, remove_pixels_h, 0)),
            SqueezeTransform(),
        ])
    def __call__(self, sample):
        return self.transform(sample)


if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args()
    main(args)
