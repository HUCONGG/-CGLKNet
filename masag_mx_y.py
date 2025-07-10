import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import functools
import math
import timm
from timm.models.layers import DropPath, to_2tuple
import einops
from fvcore.nn import FlopCountAnalysis


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)

    return x
#################
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))

#         mip = max(8, inp // reduction)

#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()

#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         identity = x

#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)

#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)

#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)

#         a_h = self.conv_h(x_h)
#         a_w = self.conv_w(x_w)

#         # out = identity * a_w * a_h 
#         # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

#         return a_h , a_w 

#################
class SimpleContrastEnhancement(nn.Module):
    def __init__(self, num_channels):
        super(SimpleContrastEnhancement, self).__init__()
        # 可学习的缩放因子（gamma）和偏移因子（beta）
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # 计算输入特征的均值和标准差
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        
        # 进行标准化处理
        x_normalized = (x - mean) / (std + 1e-5)  # 1e-5用于防止除以零
        
        # 使用gamma和beta进行对比度调整
        x_enhanced = x_normalized * self.gamma + self.beta
        
        return x_enhanced
#################
class MSCAA(nn.Module):
    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):

        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(3*channels, channels, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        # attn_0 = self.conv0_1(attn)
        # # print(attn_0.shape)
        # attn_0 = self.conv0_2(attn_0)
        # # print(attn_0.shape)
        # attn_1 = self.conv1_1(attn)
        # attn_1 = self.conv1_2(attn_1)
        # # print(attn_1.shape)
        # attn_2 = self.conv2_1(attn)
        # attn_2 = self.conv2_2(attn_2)
        ####修改
        attn_0_x = self.conv0_1(attn)
        # print(attn_0.shape)
        attn_0_y = self.conv0_2(attn)
        attn_0 = attn_0_x * attn_0_y
        # print(attn_0.shape)
        attn_1_x = self.conv1_1(attn_0_x)
        attn_1_y = self.conv1_2(attn_0_y)
        attn_1 = attn_1_x * attn_1_y
        # print(attn_1.shape)
        attn_2_x = self.conv2_1(attn_1_x)
        attn_2_y = self.conv2_2(attn_1_y)
        attn_2 = attn_2_x * attn_2_y
        attn7_0, attn11_1, attn21_2 = attn_0, attn_1, attn_2
        return attn7_0, attn11_1, attn21_2
       
#################
class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    # self.mx_y1 = MSCAA(dim)
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)
    #################
    # attn7_0, attn11_1, attn21_2 = self.mx_y1(x)
    # #################
    # x_g = attn7_0*g + attn11_1*g + attn21_2*g 
    fuse = self.bn(x+g)
    return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))
    self.gap  = nn.AdaptiveAvgPool2d((1,1))
    
  def forward(self,x,g):
      ### Option 1 ###多尺度空间特征融合
    x_ = x.clone()
    g_ = g.clone()

    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

    multi = self.multi(x, g) # B, C, H, W

  
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
      ### Option 2 ###多尺度空间特征融合
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_
    x_att = x_att + x_
    g_att = g_att + g_
    batch_size, num_channels, height, width = x_att.size()
    # feat = torch.cat((x_att, g_att), dim = 1)
    x_c_att = self.gap(x_att)
    g_c_att = self.gap(g_att)
    
    attn = torch.cat((x_c_att, g_c_att), dim = 1)
    attn = attn.view(batch_size,2, num_channels, 1, 1)

    attn = F.softmax(attn, dim = 1)
    xg_sig = torch.sigmoid(x_att + g_att)
    ####option 3 ####细化边界引导注意力
    x_att = xg_sig * x_att
    g_att = xg_sig * g_att
    
    interaction =torch.cat((x_att, g_att), dim = 1)
    interaction = interaction.view(batch_size, 2,num_channels, height, width)
    interaction = torch.sum(interaction * attn, dim = 1)

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    #y = self.bn_2(weighted + y)
    y = self.bn_2(y)
    # print('mulit')
    return y
class MultiScaleGatedAttn_GBM_ng(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    # self.Coord1 =CoordAtt(dim,dim)
    # self.Coord2 =CoordAtt(dim,dim)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))
    self.gap  = nn.AdaptiveAvgPool2d((1,1))
    
  def forward(self,x,g):
      ### Option 1 ###多尺度空间特征融合
    x_ = x.clone()
    g_ = g.clone()

    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

    multi = self.multi(x, g) # B, C, H, W

  
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
      ### Option 2 ###多尺度空间特征融合
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_
    x_att = x_att + x_
    g_att = g_att + g_
    batch_size, num_channels, height, width = x_att.size()
    # feat = torch.cat((x_att, g_att), dim = 1)
    x_c_att = self.gap(x_att)
    g_c_att = self.gap(g_att)
    
    attn = torch.cat((x_c_att, g_c_att), dim = 1)
    attn = attn.view(batch_size,2, num_channels, 1, 1)

    attn = F.softmax(attn, dim = 1)
    xg_sig = torch.sigmoid(x_att + g_att)
    ####option 3 ####细化边界引导注意力
    x_att = xg_sig * x_att
    g_att = xg_sig * g_att
    
    interaction =torch.cat((x_att, g_att), dim = 1)
    interaction = interaction.view(batch_size, 2,num_channels, height, width)
    interaction = torch.sum(interaction * attn, dim = 1)

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    #y = self.bn_2(weighted + y)
    y = self.bn_2(y)
    # print('mulit')
    return y
#################
class MultiScaleGatedAttn_no(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.conv1 = nn.Conv2d(dim, dim, 3, padding = 1,groups= dim)
    self.bn = nn.BatchNorm2d(dim)
    self.conv2 = nn.Conv2d(dim, dim, 3, padding = 1,groups = dim)
    self.bn = nn.BatchNorm2d(dim)
    self.conv_block = nn.Conv2d(dim, dim, 1)
  def forward(self,x,g):
    # x_ = x.clone()
    # g_ = g.clone()
    x = self.conv1(x)
    g = self.conv1(g)
    x = self.bn(x)
    g = self.bn(g)
    y = x + g
    y = self.conv2(y)
    return y

class MultiScaleGatedAttn_no_Cascade(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.conv1 = nn.Conv2d(dim, dim, 3, padding = 1,groups= dim)
    self.bn = nn.BatchNorm2d(dim)
    self.conv_block = nn.Conv2d(dim, dim, 1)
  def forward(self,x,g):
    # x_ = x.clone()
    # g_ = g.clone()
    x = self.conv1(x)
    x = self.bn(x)
    y = self.conv_block(x)
    return y
if __name__ == "__main__":
    xi = torch.randn(1, 192, 28, 28).cuda()
    #xi_1 = torch.randn(1, 384, 14, 14)
    g = torch.randn(1, 192, 28, 28).cuda()
    #ff = ContextBridge(dim=192)
    # cr = CoordAtt(192,192)
    # cr.cuda()
    # h,w = cr(xi)
    # print(h.shape)
    # print(w.shape)
    # t =h*w
    # print(t.shape)
    attn = MultiScaleGatedAttn_no(dim = xi.shape[1]).cuda()

    print(attn(xi, g).shape)
