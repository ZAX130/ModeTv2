import torch
import torch.nn as nn
import torch.nn.functional as nnf
from functional import modetqkrpb_cu
'''
Modified and tested by:
Haiqiao Wang
1807903986@qq.com
Shenzhen University

If you find this code useful, please cite the paper
@article{WANG2025103862,
      title = {ModeTv2: GPU-accelerated Motion Decomposition Transformer for Pairwise Optimization in Medical Image Registration},
      journal = {Medical Image Analysis},
      pages = {103862},
      year = {2025},
      issn = {1361-8415},
      doi = {https://doi.org/10.1016/j.media.2025.103862},
      author = {Haiqiao Wang and Zhuoyuan Wang and Dong Ni and Yi Wang},
}
'''
class Corr3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(Corr3D, self).__init__()
        self.kernel_size = kernel_size
    def forward(self, q, k):  # q, k shape: B, C, H, W, T

        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        B, C, H, W, T = q.shape
        scale = 1/C

        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.reshape(B, 1, C, H+pd,W+pd,T+pd).permute(0, 1, 3, 4, 5, 2)  # 1,heads,H+2,W+2,T+2,dims
        q = q.reshape(B, 1, C, H, W, T).permute(0, 1, 3, 4, 5, 2) * scale

        corr = modetqkrpb_cu(q, k,None)# B h H W T num_tokens
        corr = corr.permute(0, 1, 5, 2, 3, 4).reshape(B, -1, H, W, T)

        return corr  # correlation shape: B, k^3, H, W, T
