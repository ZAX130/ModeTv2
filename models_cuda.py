'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Haiqiao Wang
2110246069@email.szu.edu.cn
Shenzhen University
'''
import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
from functional import modetqkrpb_cu


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, 2*c),
            ConvInsBlock(2*c, 2*c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3, out4

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat


class RegHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(RegHead, self).__init__()

        self.conv3 = nn.Conv3d(in_channels, 3, 3, 1, 1)

        self.conv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3.weight.shape))
        self.conv3.bias = nn.Parameter(torch.zeros(self.conv3.bias.shape))


    def forward(self, x):
        x = self.conv3(x)
        return x

class ModeT_cuda(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, qk_scale=None, use_rpb=True):
        super().__init__()


        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size, self.rpb_size, self.rpb_size))
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        v = grid.reshape(self.kernel_size**3, 3)
        self.register_buffer('v', v)

    def forward(self, q, k):

        B, H, W, T, C = q.shape

        q = q.reshape(B, H, W, T, self.num_heads, C // self.num_heads).permute(0,4,1,2,3,5) * self.scale  #1,heads,H,W,T,dims
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # 1, C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.reshape(B, self.num_heads, C // self.num_heads, H+pd,W+pd,T+pd).permute(0, 1, 3, 4, 5, 2) # 1,heads,H+2,W+2,T+2,dims
        attn = modetqkrpb_cu(q,k,self.rpb)
        attn = attn.softmax(dim=-1)  # B h H W T num_tokens
        x = (attn @ self.v)  # B x N x heads x 1 x 3
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(B, -1, H, W, T)

        return x

class ModeTv2_model(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_channel=1,
                 channels=4,
                 head_dim=6,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=1):
        super(ModeTv2_model, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.peblock1 = ProjectionLayer(2*c, dim=head_dim*num_heads[4])
        self.mdt1 = ModeT_cuda(head_dim*num_heads[4], num_heads[4], qk_scale=scale)
        self.reghead1 = RegHead(3 * num_heads[4], 3 * 2 * num_heads[4])

        self.peblock2 = ProjectionLayer(4*c, dim=head_dim*num_heads[3])
        self.mdt2 = ModeT_cuda(head_dim*num_heads[3], num_heads[3], qk_scale=scale)
        self.reghead2 = RegHead(3 * num_heads[3], 3 * 2 * num_heads[3])

        self.peblock3 = ProjectionLayer(8*c, dim=head_dim*num_heads[2])
        self.mdt3 = ModeT_cuda(head_dim*num_heads[2], num_heads[2], qk_scale=scale)
        self.reghead3 = RegHead(3 * num_heads[2], 3 * num_heads[2] * 2)

        self.peblock4 = ProjectionLayer(16*c, dim=head_dim*num_heads[1])
        self.mdt4 = ModeT_cuda(head_dim*num_heads[1], num_heads[1], qk_scale=scale)
        self.reghead4 = RegHead(3 * num_heads[1], 3 * num_heads[1] * 2)

        self.peblock5 = ProjectionLayer(32*c, dim=head_dim*num_heads[0])
        self.mdt5 = ModeT_cuda(head_dim*num_heads[0], num_heads[0], qk_scale=scale)
        self.reghead5 = RegHead(3*num_heads[0], 3*num_heads[0]*2)

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

    def forward(self, moving, fixed):

        # encoding stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        # flow estimating stage
        q5, k5 = self.peblock5(F5), self.peblock5(M5)
        w = self.mdt5(q5, k5)
        w = self.reghead5(w)
        flow = self.upsample_trilin(2*w)

        M4 = self.transformer[3](M4, flow)
        q4,k4 = self.peblock4(F4), self.peblock4(M4)
        w=self.mdt4(q4, k4)
        w = self.reghead4(w)
        flow = self.upsample_trilin(2 *(self.transformer[3](flow, w)+w))

        M3 = self.transformer[2](M3, flow)
        q3, k3 = self.peblock3(F3), self.peblock3(M3)
        w = self.mdt3(q3, k3)
        w = self.reghead3(w)
        flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

        M2 = self.transformer[1](M2, flow)
        q2,k2 = self.peblock2(F2), self.peblock2(M2)
        w=self.mdt2(q2, k2)
        w = self.reghead2(w)
        flow = self.upsample_trilin(2 *(self.transformer[1](flow, w)+w))

        M1 = self.transformer[0](M1, flow)
        q1, k1 = self.peblock1(F1), self.peblock1(M1)
        w=self.mdt1(q1, k1)
        w=self.reghead1(w)
        flow = self.transformer[0](flow, w)+w

        y_moved = self.transformer[0](moving, flow)

        return y_moved, flow

if __name__ == '__main__':
#     # model = VoxResNet().cuda()
#     # A = torch.ones((1,1,160,196,160))
#     # B = torch.ones((1,1,160,196,160))
#     # output1 = model(A.cuda())
#     # output2 = model(B.cuda())
#     # for i in range(len(output2)):
#     #     print(torch.sum(output1[i]==output2[i]).item())
#     #     print(output1[i].shape[0]*output1[i].shape[1]*output1[i].shape[2]*output1[i].shape[3]*output1[i].shape[4])
    inshape = (1, 1,160, 160,192)
    torch.cuda.set_device(2)
    model = ModeTv2_model(inshape[2:]).cuda()
    # print(str(model))
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    out, flow = model(A.cuda(), B.cuda())
    print(out.shape, flow.shape)