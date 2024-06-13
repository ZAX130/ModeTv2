# ModeTv2: GPU-accelerated Motion Decomposition Transformer for Pairwise Optimization in Medical Image Registration

By Haiqiao Wang, Zhuoyuan Wang, Dong Ni, Yi Wang.

Paper link: [[arxiv]](https://arxiv.org/abs/2403.16526)

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-11.3-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

![图片1](https://github.com/ZAX130/ModeTv2/assets/43944700/b594621a-07c5-4eb3-8ac8-c5c9f2315499)

ModeTv1(ModeT) links： [[paper]](https://github.com/ZAX130/SmileCode)  [[code]](https://github.com/ZAX130/SmileCode)
## Dataset
The official access addresses of the public data sets are as follows：

LPBA [[link]](https://resource.loni.usc.edu/resources/atlases-downloads/)

Mindboggle [[link]](https://osf.io/yhkde/)

ABCT [[link]](https://cloud.imi.uni-luebeck.de/s/yiQZfo43YBBg7zL)

## Instructions and System Environment
The way of installation of modet package:

`cd modet`

`pip install .`

Please note that the `modet` package requires CUDA to be installed in the system's environment rather than the `cudatoolkit` package from the conda environment. If the `cudatoolkit` package is already present in the environment, please ensure it matches the system's CUDA version.

Our successfully installed environment for modet is as follows:
- Ubuntu 22.04
- pip 21.2.4
- gcc 9.5.0
- CUDA 11.3
- Python 3.9
- PyTorch 1.11.0
- NumPy 1.21.5
- Nvidia Tesla V100

For convenience, we are sharing the preprocessed [LPBA](https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0) dataset used in our experiments. Once uncompressed, simply modify the "LPBA_path" in `train.py` to the path name of the extracted data. Next, you can execute `train.py` to train the network, and after training, you can run `infer.py` to test the network performance. The small version of ModeTv2 and ModeTv2-diff can run on 2080ti on our preprocessed LPBA dataset. Please note that the suffix "_diff" denotes the diffeomorphic model.

## Citation
If you use the code in your research, please cite:
```
@misc{wang2024modetv2,
      title={ModeTv2: GPU-accelerated Motion Decomposition Transformer for Pairwise Optimization in Medical Image Registration}, 
      author={Haiqiao Wang and Zhuoyuan Wang and Dong Ni and Yi Wang},
      year={2024},
      eprint={2403.16526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph). We are very grateful for their contributions. The file makePklDataset.py shows how to make a pkl dataset from the original LPBA dataset. If you have any other questions about the .pkl format, please refer to the github page of [[TransMorph_on_IXI]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). 

## Baseline Methods
Several PyTorch implementations of some baseline methods can be found at [[SmileCode]](https://github.com/ZAX130/SmileCode/tree/main).

## How can other datasets be used in this code?
This is a common question, and please refer to the github page of [ChangeDataset.md](https://github.com/ZAX130/ModeTv2/blob/main/ChangeDataset.md) for more information.
