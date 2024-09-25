import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import io
from typing import Tuple


def setup_model(device: torch.device) -> Tuple[torch.nn.Module, int]:
    image_size = 384
    model = torch.hub.load('alexsax/omnidata_models', 'depth_dpt_hybrid_384')
    model.to(device)
    model.eval()

    return model, image_size

def setup_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, image_size = setup_model(device)
trans_totensor = setup_transforms(image_size)

def estimate_depth(input_image: PIL.Image.Image) -> PIL.Image.Image:
    with torch.no_grad():
        img_tensor = trans_totensor(input_image)[:3].unsqueeze(0).to(device)
        
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)

        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
        output = 1 - output.clamp(0, 1)

        # Convert to colormap
        plt.figure(figsize=(10, 10))
        plt.imshow(output[0].cpu().numpy(), cmap='viridis')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        output_image = Image.open(buf)
        plt.close()

    return output_image

iface = gr.Interface(
    fn=estimate_depth,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Monocular Depth Estimation: Omnidata DPT-Hybrid",
    description="Upload an image to estimate monocular depth. To use these models locally, you can use `torch.hub.load`. Code and examples in our [Github](https://github.com/alexsax/omnidata_models) repository. More information and the paper in the project page [Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans](https://omnidata.epfl.ch/).",
    examples=[
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/test1_rgb.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test2.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test3.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test4.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test5.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test6.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test7.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test8.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test9.png?raw=true",
        "https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/assets/demo/test10.png?raw=true",
    ],
)

if __name__ == "__main__":
    iface.launch()
