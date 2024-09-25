"""
Example usage:
python demo.py --task normal --img_path assets/demo/test1.png --output_path assets/ --root_dir ./pretrained_models/

This script processes images for depth estimation or surface normal prediction using Omnidata models.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import glob
import sys
from typing import Tuple, List, Optional, Union
import requests
from io import BytesIO

from omnidata_models.model.unet import UNet
from omnidata_models.model.dpt.dpt_depth import DPTDepthModel

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')
    parser.add_argument('--task', dest='task', help="normal or depth", default='normal')
    parser.add_argument('--img_path', dest='img_path', help="path to rgb image", default=None)
    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored", default='output/')
    parser.add_argument('--root_dir', dest='root_dir', help="path to pretrained models", default=None)
    return parser.parse_args()

def download_default_image() -> PIL.Image.Image:
    url = "https://github.com/EPFL-VILAB/omnidata/raw/main/omnidata_tools/torch/assets/test1_rgb.png"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception("Failed to download the default image")

def setup_model(task: str, root_dir: Optional[str] = None, device: torch.device = 'cpu') -> Tuple[torch.nn.Module, int]:
    if task == 'normal':
        image_size = 384
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
        hf_url = 'https://huggingface.co/sashasax/omnidata_normal_dpt_hybrid_384/resolve/main/omnidata_normal_dpt_hybrid.pth'
    elif task == 'depth':
        image_size = 384
        model = DPTDepthModel(backbone='vitb_rn50_384')
        hf_url = 'https://huggingface.co/sashasax/omnidata_depth_dpt_hybrid_384/resolve/main/omnidata_depth_dpt_hybrid.pth'
    else:
        raise ValueError("Task should be one of the following: normal, depth")

    if root_dir is None:
        print(f"Downloading {task} model from HuggingFace...")
        state_dict = torch.hub.load_state_dict_from_url(hf_url, map_location=device, weights_only=True)
    else:
        checkpoint_path = os.path.join(root_dir, f'omnidata_{task}_dpt_hybrid.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)

    return model, image_size

def setup_transforms(task: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    trans_totensor = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5) if task == 'depth' else lambda x: x
    ])

    trans_rgb = transforms.Compose([
        transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(512)
    ])

    return trans_totensor, trans_rgb

def standardize_depth_map(img: torch.Tensor, mask_valid: torch.Tensor = None, trunc_value: float = 0.1) -> torch.Tensor:
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean, trunc_var = trunc_img.mean(), trunc_img.var()
    img = torch.nan_to_num(img, nan=trunc_mean)
    return (img - trunc_mean) / torch.sqrt(trunc_var + 1e-6)

def save_outputs(img_input: Union[str, PIL.Image.Image], output_file_name: str, args: argparse.Namespace, model: torch.nn.Module, 
                 trans_totensor: transforms.Compose, trans_rgb: transforms.Compose, device: torch.device) -> None:
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')
        print(f'Processing input...')
        
        if isinstance(img_input, str):
            img = Image.open(img_input)
        else:
            img = img_input
        
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = 1 - output.clamp(0, 1)
            plt.imsave(save_path, output.detach().cpu().squeeze(), cmap='viridis')
        else:
            transforms.ToPILImage()(output[0]).save(save_path)

        print(f'Writing output {save_path} ...')

def process_images(args: argparse.Namespace, model: torch.nn.Module, trans_totensor: transforms.Compose, 
                   trans_rgb: transforms.Compose, device: torch.device) -> None:
    if args.img_path is None:
        print("No image path specified. Downloading default image...")
        img = download_default_image()
        output_file_name = "default_image"
        save_outputs(img, output_file_name, args, model, trans_totensor, trans_rgb, device)
    else:
        img_path = Path(args.img_path)
        if img_path.is_file():
            save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0], args, model, trans_totensor, trans_rgb, device)
        elif img_path.is_dir():
            for f in glob.glob(os.path.join(args.img_path, '*')):
                save_outputs(f, os.path.splitext(os.path.basename(f))[0], args, model, trans_totensor, trans_rgb, device)
        else:
            print("Invalid file path!")
            sys.exit()

def main() -> None:
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_path, exist_ok=True)

    root_dir = args.root_dir if args.root_dir != 'NONE' else None
    model, image_size = setup_model(args.task, root_dir, device)
    trans_totensor, trans_rgb = setup_transforms(args.task, image_size)

    process_images(args, model, trans_totensor, trans_rgb, device)

if __name__ == '__main__':
    main()