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
    if v.lower() in ('true', '1'):
        return True

    elif v.lower() in ('false', '0'):
        return False
        
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
    
    parser.add_argument("--simim_bottleneck", type = int, default =1, help='simMIM model param: bottleneck')
    parser.add_argument("--simim_depth", type = int, default =12, help='simMIM model param: depth')
    parser.add_argument("--simim_mlp_ratio", type = int, default =4, help='simMIM model param: mlp ratio')
    parser.add_argument("--simim_num_heads", type = int, default =6, help='simMIM model param: number heads')
    parser.add_argument("--simim_emb_dim", type = int, default =384, help='simMIM model param: embedding dim')
    parser.add_argument("--simim_encoder_stride", type = int, default =16, help='simMIM model param: encoder stride')
    parser.add_argument("--simim_in_chans", type = int, default =3, help='simMIM model param: depth')
    parser.add_argument("--simim_use_bn", type = str2bool, default =True, help='simMIM model param: use batch normalization')
    parser.add_argument("--simim_patch_size", type = int, default =16, help='simMIM model param: patch size')
    parser.add_argument("--simim_mask_patch_size", type = int, default =32, help='simMIM data augmentation param: mask patch size')
    parser.add_argument("--simim_mask_ratio", type = float, default =0.5, help='simMIM data augmentation param: mask ratio')
    parser.add_argument("--simim_drop_path_rate", type = float, default =0.1, help='simMIM data augmentation param: drop path rate')

    parser.add_argument('--mae_input_size', default=224, type=int,
                    help='images input size')

    return parser


def generate_saliency_map(model, image, target_value, device):  # target_value is a float
    image = image.unsqueeze(0).to(device).requires_grad_()
    target = torch.tensor([[target_value]], dtype=torch.float32, device=device)  # ✅ FIXED
    output = model(image)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    saliency = image.grad.abs().squeeze().cpu().numpy()
    return saliency.max(axis=0)  # max over channels


def plot_image_saliency_overlay(image, saliency, pred, true, out_path):
    if image.shape[0] == 3:
        image = np.moveaxis(image.cpu().numpy(), 0, -1)
        image = np.mean(image, axis=-1)
    else:
        image = image.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(saliency, cmap='hot')
    axs[1].set_title('Saliency Map')
    axs[1].axis('off')

    axs[2].imshow(image, cmap='gray')
    axs[2].imshow(saliency, cmap='hot', alpha=0.5)
    axs[2].set_title('Overlay')
    axs[2].axis('off')

    # 👇 Set overall figure title
    fig.suptitle(f'Pred: {pred:.1f} | True: {true:.1f}', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.savefig(out_path, dpi=300)
    plt.close()



def main(args):
    device = torch.device(args.device)
    general_utils.set_reproducibility(args.seed)

    args.pretrained_path = args.exp_dir / 'best_finetuned.pth'
    args.task = 'regression'
    args.num_classes = 1

    print(f"Loading backbone: {args.backbone}")

    if ('beit' in args.backbone) or ('simim' in str(args.pretrained_path)):
        print(f"📦 Loading BEiT model and head from {args.pretrained_path}")
        
        # 1. Instantiate BEiT backbone
        backbone = beit_small(args)  # will already match the expected args.simim_emb_dim
        backbone.head = None
        backbone.to(device)
        backbone.eval()

        # 2. Create matching head (based on finetuning logic)
        head = nn.Linear(args.simim_emb_dim, 1)
        head.to(device)
        head.eval()

        # 3. Load weights from saved checkpoint
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        msg_backbone = backbone.load_state_dict(checkpoint['backbone'], strict=True)
        msg_head = head.load_state_dict(checkpoint['head'], strict=True)
        print(f'✅ Loaded BEiT backbone msg: {msg_backbone}')
        print(f'✅ Loaded head msg: {msg_head}')

        # 4. Combine into a wrapper model for interpretability
        class BeitWithHead(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                x = self.backbone.forward_blocks(x)
                x = self.head(x)
                return x

        model = BeitWithHead(backbone, head).to(device)
    else:
        model = deit_vision_transformer.__dict__[args.backbone](
            num_classes=args.num_classes,
            global_pool=False
        )

        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict, strict=True)
        print(f'✅ Loaded fine-tuned model with msg: {msg}')


    model.to(device)
    model.eval()

    print("Loading validation data and tabular scaler...")
    _, indexes_val_folds, _, targets_val_folds, _, tabular_scaler_folds, _ = dataset_utils.bootstrap(args)
    val_indexes = indexes_val_folds[0]
    val_targets = targets_val_folds[0]
    print('val targets from loader: ', val_targets)
    scaler = tabular_scaler_folds[0]

    transform = [ValTransform_Resize]
    val_dataset = dataset_utils.ADNI_AGE_Dataset(args, val_targets, val_indexes, transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ✅ Double check mapping
    print("📋 Full sample_id -> target mapping:")
    for i in range(len(val_dataset)):
        _, _, target, _, sample_id = val_dataset[i]
        true_val = scaler.inverse_transform(np.array([[target.item()]]))[0][0] if scaler else target.item()
        print(f"  - ID: {sample_id} | True target: {true_val:.2f}")

    output_dir = args.exp_dir / 'saliency_maps'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating saliency maps...")
    for idx, (img, _, target, _, sample_id) in enumerate(tqdm(val_loader)):
        img = img.to(torch.float32)
        target = target.view(-1, 1).to(torch.float32)

        # ✅ FIX: detach before numpy
        pred_np = model(img.to(device)).detach().cpu().numpy()
        target_np = target.cpu().numpy()

        if scaler:
            pred_inv = scaler.inverse_transform(pred_np)[0][0]
            target_inv = scaler.inverse_transform(target_np)[0][0]
        else:
            pred_inv = pred_np[0][0]
            target_inv = target_np[0][0]

        saliency = generate_saliency_map(model, img[0], target[0][0].item(), device)

        sample_num = int(sample_id[0])
        print(f"[{idx}] ID: {sample_num} | True target: {target_inv:.2f} | Pred: {pred_inv:.2f}")

        delta = abs(pred_inv - target_inv)
        filename = f"delta_{delta:.1f}_sample{sample_num}_pred{pred_inv:.1f}_true{target_inv:.1f}.png"

        out_path = output_dir / filename
        plot_image_saliency_overlay(img[0], saliency, pred_inv, target_inv, out_path)

    print(f"✅ Saliency maps saved in {output_dir}")


# --- Helper Transforms ---
class UnsqueezeTransform(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return torch.unsqueeze(tensor, self.dim)


class SqueezeTransform(object):
    def __call__(self, tensor):
        return torch.squeeze(tensor)


class ValTransform_Resize(object):
    def __init__(self, args, train_set_mean, train_set_std):
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
