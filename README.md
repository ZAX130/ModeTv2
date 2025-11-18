# ModeTv2: GPU-accelerated Motion Decomposition Transformer for Pairwise Optimization in Medical Image Registration (MIA2026)

By Haiqiao Wang, Zhuoyuan Wang, Dong Ni, Yi Wang.

Paper link: [[paper]](https://doi.org/10.1016/j.media.2025.103862) [[arxiv]](https://arxiv.org/abs/2403.16526)
## News
(06/11/2025) &#127881; The paper has been accepted by Medical Image Analysis.

(28/12/2024) We implemented the CUDA version of the 3D Correlation layer based on the modet package. For specific usage, see [Correlation&modet.md](https://github.com/ZAX130/ModeTv2/blob/main/Corr3D%26modet.md)

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-11.3-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

![图片1](https://github.com/ZAX130/ModeTv2/assets/43944700/b594621a-07c5-4eb3-8ac8-c5c9f2315499)

ModeTv1(ModeT) links： [[paper]](https://github.com/ZAX130/SmileCode)  [[code]](https://github.com/ZAX130/SmileCode)
## Dataset
The access addresses of the official and preprocessed public data sets are as follows：

LPBA [[link]](https://resource.loni.usc.edu/resources/atlases-downloads/) [[preprocessed]](https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0)

Mindboggle [[link]](https://osf.io/yhkde/) [[preprocessed]](https://drive.usercontent.google.com/download?id=17WLZ-eJwNTG0U93WuEuyWUOQlhsUrxjJ&export=download&authuser=0) 

ABCT [[link]](https://cloud.imi.uni-luebeck.de/s/yiQZfo43YBBg7zL) [[preprocessed]](https://drive.usercontent.google.com/download?id=1hrb-qVgbF1acZ5V-tGrvbl2CUHi3dAZP&export=download&authuser=0)

IXI [[link]](https://surfer.nmr.mgh.harvard.edu/pub/data/) [[preprocessed]](https://drive.usercontent.google.com/download?id=1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol&export=download&authuser=0)

Note that the preprocessed version of IXI is adopted from [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md).
## Instructions and System Environment
The way of installation of modet package:

`cd modet`

`pip install .`

Please note that the `modet` package requires CUDA to be installed in the system's environment rather than the `cudatoolkit` package from the conda environment. If the `cudatoolkit` package is already present in the environment, please ensure it matches the system's CUDA version.

Our successfully installed environment for modet is as follows:
- Ubuntu 22.04 / Windows 11
- pip 21.2.4
- gcc 9.5.0 / MSVC v142 (VS2022)
- CUDA 11.3/11.8/12.1
- Python 3.9/3.11/3.12
- PyTorch 1.11.0/2.3.0/2.4.1
- NumPy 1.21.5
- Nvidia Tesla V100/Nvidia RTX 2080Ti/Nvidia RTX 3090

For convenience, we use the preprocessed [LPBA](https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0) dataset as an example. Once uncompressed, simply modify the "LPBA_path" in `train.py` to the path name of the extracted data. Next, you can execute `train.py` to train the network, and after training, you can run `infer.py` to test the network performance. The small version of ModeTv2 and ModeTv2-diff can run on 2080ti on our preprocessed LPBA dataset. Please note that the suffix "_diff" denotes the diffeomorphic model.

## Citation
If you find the code useful, please cite our paper.
```
@article{WANG2025103862,
title = {ModeTv2: GPU-accelerated Motion Decomposition Transformer for Pairwise Optimization in Medical Image Registration},
journal = {Medical Image Analysis},
pages = {103862},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103862},
author = {Haiqiao Wang and Zhuoyuan Wang and Dong Ni and Yi Wang},
}
```
The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph). We are very grateful for their contributions. The file makePklDataset.py shows how to make a pkl dataset from the original LPBA dataset. If you have any other questions about the .pkl format, please refer to the github page of [[TransMorph_on_IXI]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). 

## Baseline Methods
Several PyTorch implementations of some baseline methods can be found at [[SmileCode]](https://github.com/ZAX130/SmileCode/tree/main).

## How can other datasets be used in this code?
This is a common question, and please refer to the github page of [ChangeDataset.md](https://github.com/ZAX130/ModeTv2/blob/main/ChangeDataset.md) for more information.
