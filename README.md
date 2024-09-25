<div align="center">


# Omnidata: _PyTorch Models and Dataloaders_
[`Project Website`](https://omnidata.vision) &centerdot; [`Paper`](https://arxiv.org/abs/2110.04994) &centerdot; [`Github`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_tools/torch) &centerdot; [`Data`](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/dataset#readme) &centerdot; [`PyTorch Utils + Weights`](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#readme) &centerdot;  [`Annotator`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_annotator#readme) 

**Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans (ICCV 2021)**

</div>


---

This repository contains pretrained models for surface normal estimation and depth estimation from the Omnidata project. It integrates with `torch.hub` for easy model loading and includes a simple Gradio demo.

#### Use pretrained models 

```
import torch

# Load surface normal estimation model
model_normal = torch.hub.load('alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384')

# Load depth estimation model
model_depth = torch.hub.load('alexsax/omnidata_models', 'depth_dpt_hybrid_384')

# Load a model without pre-trained weights
model_custom = torch.hub.load('alexsax/omnidata_models', 'dpt_hybrid_384', pretrained=False, task='normal')
```

The checkpoints are hosted on Hugging Face:
- [Surface Normal DPT-Hybrid 384](https://huggingface.co/sashasax/omnidata_normal_dpt_hybrid_384)
- [Depth DPT-Hybrid 384](https://huggingface.co/sashasax/omnidata_depth_dpt_hybrid_384)

## Run gradio demo
This will launch a web interface where you can upload images and see the estimated surface normals.

```bash
python gradio_app.py
```

## Run script locally
You can run them on your own image with the following command:
```bash
python demo.py --task $TASK --img_path $PATH_TO_IMAGE_OR_FOLDER --output_path $PATH_TO_SAVE_OUTPUT 
```
The `--task` flag should be either `normal` or `depth`. To run the script for a `normal` target on an [example image](./assets/demo/test1.png):
```bash
python demo.py --task normal --img_path assets/demo/test1.png --output_path assets/
```
|  |   |   |   |  |  |  |
| :-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| ![](./assets/demo/test1.png) | ![](./assets/demo/test2.png) |![](./assets/demo/test3.png) | ![](./assets/demo/test4.png) | ![](./assets/demo/test5.png) |![](./assets/demo/test7.png) |![](./assets/demo/test9.png) |
| ![](./assets/demo/test1_normal.png) | ![](./assets/demo/test2_normal.png) |![](./assets/demo/test3_normal.png) | ![](./assets/demo/test4_normal.png) | ![](./assets/demo/test5_normal.png) | ![](./assets/demo/test7_normal.png) | ![](./assets/demo/test9_normal.png) |
| ![](./assets/demo/test1_depth.png) | ![](./assets/demo/test2_depth.png) | ![](./assets/demo/test3_depth.png) | ![](./assets/demo/test4_depth.png) | ![](./assets/demo/test5_depth.png) | ![](./assets/demo/test7_depth.png) | ![](./assets/demo/test9_depth.png)



### Network Architecture
- **Version 2 models** _(stronger than V1)_ **[March 2022]**: <br> These are DPT architectures trained on more data using both [3D Data Augmentations](https://3dcommoncorruptions.epfl.ch/) and [Cross-Task Consistency](https://consistency.epfl.ch/). Here's the list of updates in Version 2 models:
  - **Monocular Depth Estimation:**
    - [Habitat-Matterport 3D Dataset (HM3D)](https://aihabitat.org/datasets/hm3d/) and 5 [MiDaS](https://github.com/isl-org/MiDaS) dataset components (RedWebDataset, HRWSIDataset, MegaDepthDataset, TartanAirDataset, BlendedMVS) are added to the training data.
    - 1 week of training with 2D and [3D data augmentations](https://3dcommoncorruptions.epfl.ch/) and 1 week of training with [cross-task consistency](https://consistency.epfl.ch/) on 4xV100.
  - **Monocular Surface Normal Estimation:**
    - New model is based on DPT architecture.
    - Habitat-Matterport 3D Dataset (HM3D) is added to the training data.
    - 1 week of training with 2D and [3D data augmentations](https://3dcommoncorruptions.epfl.ch/) and 1 week of training with [cross-task consistency](https://consistency.epfl.ch/) on 4xV100.


#### Packages
You can see the complete list of required packages in [requirements.txt](https://github.com/Ainaz99/omnidata-tools/blob/main/requirements.txt). We recommend using conda or virtualenv for the installation.

```bash
conda create -n testenv -y python=3.8
source activate testenv
pip install -r requirements.txt
```


## Citation
If you find the code or models useful, please cite our paper:
```
@inproceedings{eftekhar2021omnidata,
  title={Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets From 3D Scans},
  author={Eftekhar, Ainaz and Sax, Alexander and Malik, Jitendra and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10786--10796},
  year={2021}
}
```
In case you use our latest pretrained models please also cite the following paper:
```
@inproceedings{kar20223d,
  title={3D Common Corruptions and Data Augmentation},
  author={Kar, O{\u{g}}uzhan Fatih and Yeo, Teresa and Atanov, Andrei and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18963--18974},
  year={2022}
}
```
