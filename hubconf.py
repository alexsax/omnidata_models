import torch
from omnidata_models.model.dpt.dpt_depth import DPTDepthModel

dependencies = ['torch']

def dpt_hybrid_384(pretrained=True, task='normal', **kwargs):
    """
    DPT-Hybrid model for monocular depth or surface normal estimation
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on Omnidata
        task (str): 'normal' for surface normal estimation, 'depth' for depth estimation
    
    Returns:
        torch.nn.Module: DPT-Hybrid model
    """
    if task not in ['normal', 'depth']:
        raise ValueError("Task should be one of the following: normal, depth")

    if task == 'normal':
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
    else:
        model = DPTDepthModel(backbone='vitb_rn50_384')
    
    if pretrained:
        if task == 'normal':
            url = 'https://huggingface.co/sashasax/omnidata_normal_dpt_hybrid_384/resolve/main/omnidata_normal_dpt_hybrid.pth'
        else:  # depth
            url = 'https://huggingface.co/sashasax/omnidata_depth_dpt_hybrid_384/resolve/main/omnidata_depth_dpt_hybrid.pth'
        
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)
        model.load_state_dict(state_dict)
    
    return model

def surface_normal_dpt_hybrid_384(pretrained=True, **kwargs):
    """
    DPT-Hybrid model for monocular surface normal estimation
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on Omnidata
    
    Returns:
        torch.nn.Module: DPT-Hybrid model for surface normal estimation
    """
    return dpt_hybrid_384(pretrained=pretrained, task='normal', **kwargs)

def depth_dpt_hybrid_384(pretrained=True, **kwargs):
    """
    DPT-Hybrid model for monocular depth estimation
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on Omnidata
    
    Returns:
        torch.nn.Module: DPT-Hybrid model for depth estimation
    """
    return dpt_hybrid_384(pretrained=pretrained, task='depth', **kwargs)