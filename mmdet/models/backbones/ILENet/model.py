# ILENet
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import os
import math
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from torchvision.transforms import transforms

# from torch.nn.functional import interpolate, conv2d
from ...builder import BACKBONES
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import BaseModule
# from .global_net import Global_pred


from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os

import cv2

from einops import rearrange
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

# v1-44
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
# from model.blocks import CBlock_ln, SwinTransformerBlock
# from model.global_net import Global_pred
# from blocks import CBlock_ln, SwinTransformerBlock
# from global_net import Global_pred
import cv2

from einops import rearrange
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
# from model.blocks import Mlp
from PIL import Image


class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # input: [b 10 c=64]
        # output: [b 10 c=64]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 12, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 12, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=1):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.fusion = CSAF(3 * dim)

        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # input: [b c=64 H W]
        # [b c H W]
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.conv1(x_0)
        x22 = F.interpolate(y_0, scale_factor=0.5, mode='bilinear')
        y_2 = self.conv2(x_2 + x22)
        x44 = F.interpolate(y_2, scale_factor=0.5, mode='bilinear')
        y_4 = self.conv3(x_4 + x44)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        x = y + x

        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Global_pred(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4, type='lol'):
        super(Global_pred, self).__init__()
        self.gamma_base = nn.Parameter(torch.ones((1, 3, 1, 1)), requires_grad=True)
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix
        # main blocks
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv_large(x)
        x = self.generator(x)
        gamma, color = x[:, 0:3], x[:, 3:]
        gamma = self.gamma_linear(gamma).unsqueeze(-1) + self.gamma_base
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base
        return gamma, color


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class pd_conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(pd_conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                      padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class DFM(nn.Module):
    def __init__(self, in_channel, inter_num2):
        super(DFM, self).__init__()
        self.pre_conv = nn.Conv2d(in_channel, inter_num2, 3, 1, 1)
        self.relu = nn.ReLU()
        self.channel_splits = [inter_num2 // 2, inter_num2 // 2]

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.conv = nn.Conv2d(2 * inter_num2, inter_num2 // 2, 3, 1, 1)
        self.pconv = nn.Conv2d(inter_num2, in_channel, 1, 1)

    def forward(self, feats):
        feats = self.pre_conv(feats)
        fea1, fea2 = torch.split(feats, self.channel_splits, dim=1)
        fea1_1 = self.maxpool1(fea1)
        fea1_2 = self.maxpool2(fea1)
        fea1_3 = self.maxpool3(fea1)
        fea1_4 = self.maxpool4(fea1)
        fea1 = torch.cat([fea1_1, fea1_2, fea1_3, fea1_4], 1)
        fea1 = self.conv(fea1)

        fea = torch.cat([fea1, fea2], 1)

        fea = self.pconv(fea)
        return self.relu(fea)


class DDM(nn.Module):
    def __init__(self, in_channel, inter_num, pd_nums):
        super(DDM, self).__init__()
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for _ in range(pd_nums):
            dense_conv = pd_conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = nn.Conv2d(c, in_channel, 1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SEM(nn.Module):
    def __init__(self, in_channel=3, inter_channel=16, inter_channel2=32, pd_nums=3):
        super(SEM, self).__init__()
        self.ddm = DDM(in_channel, inter_channel, pd_nums)
        self.dfm = DFM(in_channel, inter_channel2)

        self.last_conv = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        x1 = self.ddm(x)
        x2 = self.dfm(x)
        x = self.last_conv(x1 + x2) + x

        return x


class DownsampleNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=40):
        super(DownsampleNet, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel // 4, out_channel // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=2, padding=1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        fea = self.downsample(x)
        out = self.gate(fea) * fea + fea

        return out


class Illumination_Estimator(nn.Module):
    def __init__(
            self, in_channel=3, out_channel=40):
        super(Illumination_Estimator, self).__init__()
        self.down = DownsampleNet(in_channel + 1, out_channel)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        illu_map = self.down(input)

        return illu_map


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim // mult, dim // mult, 3, 1, 1,
                      bias=False, groups=dim // mult),
            GELU(),
            nn.Conv2d(dim // mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            dim=40,
            dim_head=40,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channel, dim, 3, 1, 1, bias=False)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
        self.post_conv = nn.Conv2d(dim, out_channel, 3, 1, 1, bias=False)

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        可以看到 IGAB 并不会改变特征图尺寸大小
        """
        xx = self.pre_conv(x)
        xx = xx.permute(0, 2, 3, 1)  # 把通道维度调整到最后，从而得到 [b, h, w, c] 这个是可以理解的，因为通道即图上区块的特征
        for (attn, ff) in self.blocks:
            xx = attn(xx, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + xx
            # [b, h, w, c]
            xx = ff(xx) + xx
        out = xx.permute(0, 3, 1, 2)  # 再变为 [b, c, h, w]
        out = self.post_conv(out) + x
        return out


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image


class Up_guide(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


@BACKBONES.register_module()
class ILENet(nn.Module):
    def __init__(self, in_dim=3, with_global=True,
                 pretrained=None,
                 init_cfg=None,
                 num_high=3,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.global_net = Global_pred(in_channels=3, type=type)

        self.estimate = Illumination_Estimator()
        self.sem1 = SEM()
        self.igab = IGAB()
        self.up1 = Up_guide(1, 3)
        self.sem2 = SEM()
        self.up2 = Up_guide(1, 3)
        self.sem3 = SEM()
        self.up3 = Up_guide(1, 3)
        self.sem4 = SEM()

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)
        illu_map = self.estimate(x)
        t0 = pyrs[-1]
        l = self.sem1(t0)
        l = self.igab(l, illu_map)

        u1 = self.up1(l)
        t1 = pyrs[-2] + u1
        h3 = self.sem2(t1)

        u2 = self.up2(h3)
        t2 = pyrs[-3] + u2
        h2 = self.sem3(t2)

        u3 = self.up3(h2)
        t3 = pyrs[-4] + u3
        h1 = self.sem4(t3)

        trans_pyrs = [l, h3, h2, h1]
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        gamma, color = self.global_net(x)
        b = out.shape[0]
        out = out.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        out = torch.stack(
            [self.apply_color(out[i, :, :, :], color[i, :, :]) ** gamma[i, :].permute(1, 2, 0) for i in range(b)],
            dim=0)
        out = torch.clamp(out, 1e-8, 1.0 - 1e-8)
        out = out.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)

        out = out

        return out, out, out


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img = torch.Tensor(1, 3, 640, 640).cuda()
    net = ILENet().cuda()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)


