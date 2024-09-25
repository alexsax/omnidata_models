import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
import os
from typing import Tuple

from omnidata_models.model.dpt.dpt_depth import DPTDepthModel

def setup_model(device: torch.device) -> Tuple[torch.nn.Module, int]:
    image_size = 384
    # model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
    # hf_url = 'https://huggingface.co/sashasax/omnidata_normal_dpt_hybrid_384/resolve/main/omnidata_normal_dpt_hybrid.pth'
    
    # print(f"Downloading model from {hf_url}")
    # state_dict = torch.hub.load_state_dict_from_url(hf_url, map_location=device, weights_only=True)
    
    # model.load_state_dict(state_dict)
    model = torch.hub.load('alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384')
    model.to(device)
    model.eval()

    return model, image_size

def setup_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, image_size = setup_model(device)
trans_totensor = setup_transforms(image_size)

def estimate_surface_normal(input_image: PIL.Image.Image) -> PIL.Image.Image:
    with torch.no_grad():
        img_tensor = trans_totensor(input_image)[:3].unsqueeze(0).to(device)
        
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)

        output = model(img_tensor).clamp(min=0, max=1)
        output_image = transforms.ToPILImage()(output[0])

    return output_image

iface = gr.Interface(
    fn=estimate_surface_normal,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Monocular Surface Normal Estimation: Omnidata DPT-Hybrid",
    description="Upload an image to estimate monocular surface normals. To use these models locally, you can use `torch.hub.load`. Code and examples in our [Github](https://github.com/alexsax/omnidata_models) repository. More information and the paper in the project page [Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans](https://omnidata.epfl.ch/).",
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
