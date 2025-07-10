
import torch
import torch.nn as nn
import torchvision
from torchvision.ops import DeformConv2d

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class d_LSKblock(nn.Module):
    def __init__(self, dim,kernel_size=(7,7), padding=9, stride=1, dilation=3, bias=True,ration = 2):
        super().__init__()
        groups = dim
        self.dim = dim
        # channels =  2 * kernel_size[0] * kernel_size[1]
        # self.conv3X3_offset1 = nn.Conv2d(dim, 18, 3, padding=1)
        self.conv3X3_offset2 = nn.Conv2d(dim, 50, 3, padding=1,groups=2)
        self.conv3X3_offset3 = nn.Conv2d(dim, 98, 3, padding=1,groups=2)
        # # # self.conv_l0 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels,dilation=1)
        # self.conv1 = DeformConv2d(in_channels=dim, out_channels=dim, groups=dim, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True)
        # self.conv_l1 = nn.Conv2d(channels, channels, 5, stride=1, padding=6, groups=channels, dilation=3)
        self.conv2 = DeformConv2d(in_channels=dim, out_channels=dim, groups=dim, kernel_size=(5,5), padding=6, stride=1, dilation=3, bias=True)
        # self.conv3 = nn.Conv2d(channels, channels, 7,padding=9, groups=channels, dilation=3)
        self.cnov3 = DeformConv2d(in_channels=dim, out_channels=dim, groups=dim, kernel_size=(7,7), padding=9, stride=1, dilation=3, bias=True)
        # self.conv3x3 = nn.Conv2d(3*dim, 3, 3, padding=1,groups=3)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # self.conv1x1 = nn.Conv2d(3*dim, dim, 1)
        # in_channels = dim

    def forward(self, x):
        # u = x.clone()
        ################
        batch_size, dim, height, width = x.size()
        offset3 = self.conv3X3_offset3(x)
        # near_offset3 =  update_offset_with_neighborhood(offset3, kernel_size=3)
        out1 = self.cnov3(x,offset3)
        ###################
        offset2 = self.conv3X3_offset2(x)
        # near_offset2 = update_offset_with_neighborhood(offset2, kernel_size=3)
        out2 = self.conv2(out1, offset2)
        # print('d_LSKblock:')
        ##################
        # offset1 = self.conv3X3_offset1(out2)
        # near_offset1 = update_offset(offset1)
        # out3 = self.conv1(out2, near_offset1)
        ######################
        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, dim, height, width)
        ##################
        out1 = self.gap(out1)
        out2 = self.gap(out2)
        out = torch.cat([out1, out2], dim=1)
        attion  = out.view(batch_size,2,self.dim,1,1)
        softmax_attion = torch.softmax(attion, dim=1)
        out = torch.sum(feats * softmax_attion, dim=1)
        return out
#################
class d_LSKblock3x3(nn.Module):
    def __init__(self, dim, bias=True, stride=1):
        super().__init__()
        self.dim = dim

        # 第一个3x3可变形卷积及其偏移量生成器
        # 偏移量通道数: 2 * K_h * K_w = 2 * 3 * 3 = 18
        self.offset_gen1 = nn.Conv2d(dim, 18, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn1 = DeformConv2d(
            in_channels=dim,
            out_channels=dim,
            groups=dim,
            kernel_size=(3,3),
            padding=1,
            stride=stride,
            dilation=1,
            bias=bias
        )

        # 第二个3x3可变形卷积及其偏移量生成器
        self.offset_gen2 = nn.Conv2d(dim, 18, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn2 = DeformConv2d(
            in_channels=dim,
            out_channels=dim,
            groups=dim,
            kernel_size=(3,3),
            padding=1,
            stride=stride,
            dilation=1,
            bias=bias
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        offset1_val = self.offset_gen1(x)
        out1 = self.dcn1(x, offset1_val)

        offset2_val = self.offset_gen2(x) # 偏移量从输入 x 生成
        out2 = self.dcn2(out1, offset2_val)

        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, self.dim, height, width)

        att_map1 = self.gap(out1)
        att_map2 = self.gap(out2)
        
        attention_scores_cat = torch.cat([att_map1, att_map2], dim=1)
        attention_scores_reshaped = attention_scores_cat.view(batch_size, 2, self.dim, 1, 1)
        
        softmax_attention = torch.softmax(attention_scores_reshaped, dim=1)

        out = torch.sum(feats * softmax_attention, dim=1)
        
        return out
#############
class d_LSKblock5x3(nn.Module):
    # Kernel version: 5x3 for both DCNs
    def __init__(self, dim, bias=True, stride=1):
        super().__init__()
        self.dim = dim

        # DCN1: kernel (5,3)
        kernel_size1 = (5, 3)
        offset_channels1 = 2 * kernel_size1[0] * kernel_size1[1] # 2 * 5 * 3 = 30
        padding1 = ((kernel_size1[0] - 1) // 2, (kernel_size1[1] - 1) // 2) # (2, 1)

        self.offset_gen1 = nn.Conv2d(dim, offset_channels1, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn1 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size1, padding=padding1,
            stride=stride, dilation=1, bias=bias
        )

        # DCN2: kernel (5,3)
        kernel_size2 = (5, 3)
        offset_channels2 = 2 * kernel_size2[0] * kernel_size2[1] # 2 * 5 * 3 = 30
        padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2) # (2, 1)

        self.offset_gen2 = nn.Conv2d(dim, offset_channels2, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn2 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size2, padding=padding2,
            stride=stride, dilation=1, bias=bias
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        offset1_val = self.offset_gen1(x)
        out1 = self.dcn1(x, offset1_val)

        offset2_val = self.offset_gen2(x)
        out2 = self.dcn2(out1, offset2_val)

        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, self.dim, height, width)

        att_map1 = self.gap(out1)
        att_map2 = self.gap(out2)
        
        attention_scores_cat = torch.cat([att_map1, att_map2], dim=1)
        attention_scores_reshaped = attention_scores_cat.view(batch_size, 2, self.dim, 1, 1)
        
        softmax_attention = torch.softmax(attention_scores_reshaped, dim=1)

        out = torch.sum(feats * softmax_attention, dim=1)
        
        return out
############
class d_LSKblock5x5(nn.Module):
    # Kernel version: 5x5 for both DCNs
    def __init__(self, dim, bias=True, stride=1):
        super().__init__()
        self.dim = dim

        # DCN1: kernel (5,5)
        kernel_size1 = (5, 5)
        offset_channels1 = 2 * kernel_size1[0] * kernel_size1[1] # 2 * 5 * 5 = 50
        padding1 = (kernel_size1[0] - 1) // 2 # 2 (since H=W)

        self.offset_gen1 = nn.Conv2d(dim, offset_channels1, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn1 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size1, padding=padding1,
            stride=stride, dilation=1, bias=bias
        )

        # DCN2: kernel (5,5)
        kernel_size2 = (5, 5)
        offset_channels2 = 2 * kernel_size2[0] * kernel_size2[1] # 2 * 5 * 5 = 50
        padding2 = (kernel_size2[0] - 1) // 2 # 2

        self.offset_gen2 = nn.Conv2d(dim, offset_channels2, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn2 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size2, padding=padding2,
            stride=stride, dilation=1, bias=bias
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        offset1_val = self.offset_gen1(x)
        out1 = self.dcn1(x, offset1_val)

        offset2_val = self.offset_gen2(x)
        out2 = self.dcn2(out1, offset2_val)

        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, self.dim, height, width)

        att_map1 = self.gap(out1)
        att_map2 = self.gap(out2)
        
        attention_scores_cat = torch.cat([att_map1, att_map2], dim=1)
        attention_scores_reshaped = attention_scores_cat.view(batch_size, 2, self.dim, 1, 1)
        
        softmax_attention = torch.softmax(attention_scores_reshaped, dim=1)

        out = torch.sum(feats * softmax_attention, dim=1)
        
        return out
###########
class d_LSKblock9x7(nn.Module):
    # Kernel version: 9x7 for both DCNs
    def __init__(self, dim, bias=True, stride=1):
        super().__init__()
        self.dim = dim

        # DCN1: kernel (9,7)
        kernel_size1 = (9, 7)
        offset_channels1 = 2 * kernel_size1[0] * kernel_size1[1] # 2 * 9 * 7 = 126
        padding1 = ((kernel_size1[0] - 1) // 2, (kernel_size1[1] - 1) // 2) # (4, 3)

        self.offset_gen1 = nn.Conv2d(dim, offset_channels1, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn1 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size1, padding=padding1,
            stride=stride, dilation=1, bias=bias
        )

        # DCN2: kernel (9,7)
        kernel_size2 = (9, 7)
        offset_channels2 = 2 * kernel_size2[0] * kernel_size2[1] # 2 * 9 * 7 = 126
        padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2) # (4, 3)

        self.offset_gen2 = nn.Conv2d(dim, offset_channels2, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn2 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size2, padding=padding2,
            stride=stride, dilation=1, bias=bias
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        offset1_val = self.offset_gen1(x)
        out1 = self.dcn1(x, offset1_val)

        offset2_val = self.offset_gen2(x)
        out2 = self.dcn2(out1, offset2_val)

        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, self.dim, height, width)

        att_map1 = self.gap(out1)
        att_map2 = self.gap(out2)
        
        attention_scores_cat = torch.cat([att_map1, att_map2], dim=1)
        attention_scores_reshaped = attention_scores_cat.view(batch_size, 2, self.dim, 1, 1)
        
        softmax_attention = torch.softmax(attention_scores_reshaped, dim=1)

        out = torch.sum(feats * softmax_attention, dim=1)
        
        return out
###########
class d_LSKblock7x7(nn.Module):
    # Kernel version: 7x7 for both DCNs
    def __init__(self, dim, bias=True, stride=1):
        super().__init__()
        self.dim = dim

        # DCN1: kernel (7,7)
        kernel_size1 = (7, 7)
        offset_channels1 = 2 * kernel_size1[0] * kernel_size1[1] # 2 * 7 * 7 = 98
        padding1 = (kernel_size1[0] - 1) // 2 # 3

        self.offset_gen1 = nn.Conv2d(dim, offset_channels1, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn1 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size1, padding=padding1,
            stride=stride, dilation=1, bias=bias
        )

        # DCN2: kernel (7,7)
        kernel_size2 = (7, 7)
        offset_channels2 = 2 * kernel_size2[0] * kernel_size2[1] # 2 * 7 * 7 = 98
        padding2 = (kernel_size2[0] - 1) // 2 # 3

        self.offset_gen2 = nn.Conv2d(dim, offset_channels2, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn2 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size2, padding=padding2,
            stride=stride, dilation=1, bias=bias
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        offset1_val = self.offset_gen1(x)
        out1 = self.dcn1(x, offset1_val)

        offset2_val = self.offset_gen2(x)
        out2 = self.dcn2(out1, offset2_val)

        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, self.dim, height, width)

        att_map1 = self.gap(out1)
        att_map2 = self.gap(out2)
        
        attention_scores_cat = torch.cat([att_map1, att_map2], dim=1)
        attention_scores_reshaped = attention_scores_cat.view(batch_size, 2, self.dim, 1, 1)
        
        softmax_attention = torch.softmax(attention_scores_reshaped, dim=1)

        out = torch.sum(feats * softmax_attention, dim=1)
        
        return out
###########
class d_LSKblock9x9(nn.Module):
    # Kernel version: 9x9 for both DCNs
    def __init__(self, dim, bias=True, stride=1):
        super().__init__()
        self.dim = dim

        # DCN1: kernel (9,9)
        kernel_size1 = (9, 9)
        # 偏移量通道数: 2 * K_h * K_w = 2 * 9 * 9 = 162
        offset_channels1 = 2 * kernel_size1[0] * kernel_size1[1]
        # padding 保持输出尺寸不变: (kernel_size - 1) // 2 (假设 dilation=1)
        padding1 = (kernel_size1[0] - 1) // 2 # (9-1)//2 = 4

        self.offset_gen1 = nn.Conv2d(dim, offset_channels1, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn1 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size1, padding=padding1,
            stride=stride, dilation=1, bias=bias
        )

        # DCN2: kernel (9,9)
        kernel_size2 = (9, 9)
        offset_channels2 = 2 * kernel_size2[0] * kernel_size2[1] # 162
        padding2 = (kernel_size2[0] - 1) // 2 # 4

        self.offset_gen2 = nn.Conv2d(dim, offset_channels2, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.dcn2 = DeformConv2d(
            in_channels=dim, out_channels=dim, groups=dim,
            kernel_size=kernel_size2, padding=padding2,
            stride=stride, dilation=1, bias=bias
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        offset1_val = self.offset_gen1(x)
        out1 = self.dcn1(x, offset1_val)

        offset2_val = self.offset_gen2(x)
        out2 = self.dcn2(out1, offset2_val)

        feats = torch.cat([out1, out2], dim=1)
        feats = feats.view(batch_size, 2, self.dim, height, width)

        att_map1 = self.gap(out1)
        att_map2 = self.gap(out2)
        
        attention_scores_cat = torch.cat([att_map1, att_map2], dim=1)
        attention_scores_reshaped = attention_scores_cat.view(batch_size, 2, self.dim, 1, 1)
        
        softmax_attention = torch.softmax(attention_scores_reshaped, dim=1)

        out = torch.sum(feats * softmax_attention, dim=1)
        
        return out
###########
class d_LSKblock_no_edlk(nn.Module):
    def __init__(self, dim,kernel_size=(7,7), padding=9, stride=1, dilation=3, bias=True,ration = 2):
        super().__init__()
        groups = dim
        ##conv5x5
        self.conv5x5 = nn.Conv2d(dim, dim, 5,padding=6, stride=1, dilation=3, bias=True, groups=dim)
        ##conv7x7
        self.conv7x7 = nn.Conv2d(dim, dim, 7,padding=9, stride=1, dilation=3, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out1 = self.conv5x5(x)
        out2 = self.conv7x7(x)
        out = out1 + out2
        out = self.bn(out)
        out = self.relu(out)
        return out

def update_offset(offset, threshold=5.0, adjust_factor=0.5):
    """
    对输入的偏移量张量直接使用邻域差异阈值进行更新

    参数:
    offset (torch.Tensor): 输入的偏移量张量
    threshold (float): 差异阈值，默认为 5.0
    adjust_factor (float): 调整因子，默认为 0.5

    返回:
    torch.Tensor: 更新后的偏移量张量
    """
    new_offset = offset.clone()  # 复制输入的偏移量张量

    for row in range(offset.size(2)):  # 遍历行
        for col in range(offset.size(3)):  # 遍历列
            center_offset = offset[:, :, row, col]  # 获取当前位置的偏移量
            # 计算邻域偏移量
            top_offset = offset[:, :, row - 1, col] if row - 1 >= 0 else offset[:, :, row, col]
            bottom_offset = offset[:, :, row + 1, col] if row + 1 < offset.size(2) else offset[:, :, row, col]
            left_offset = offset[:, :, row, col - 1] if col - 1 >= 0 else offset[:, :, row, col]
            right_offset = offset[:, :, row, col + 1] if col + 1 < offset.size(3) else offset[:, :, row, col]
            neighborhood_mean = (top_offset + bottom_offset + left_offset + right_offset) / 4  # 计算邻域均值
            diff = torch.norm(center_offset - neighborhood_mean)  # 计算与邻域均值的差异
            if diff > threshold:  # 如果差异超过阈值
                new_offset[:, :, row, col] += (center_offset - neighborhood_mean) * adjust_factor  # 进行调整
    return new_offset
import torch
import torch.nn.functional as F

def update_offset_with_neighborhood(offset, kernel_size=5):
    """
    使用领域像素信息和卷积操作更新偏移量。

    参数:
    offset (torch.Tensor): 输入的偏移量张量 [B, num_offsets, H, W]
    kernel_size (int): 领域的尺寸

    返回:
    torch.Tensor: 更新后的偏移量张量
    """
    batch_size, num_offsets, height, width = offset.size()
    pad = kernel_size // 2
    
    # 创建均值卷积核
    kernel = torch.ones((num_offsets,num_offsets, kernel_size, kernel_size), device=offset.device) / (kernel_size * kernel_size)
    
    # 将偏移量进行填充，以处理边界情况
    offset_padded = F.pad(offset, (pad, pad, pad, pad), mode='replicate')
    
    # 使用卷积计算领域均值
    neighborhood_mean = F.conv2d(offset_padded, kernel, stride=1, padding=0)
    
    # 更新偏移量
    new_offset = offset + (offset - neighborhood_mean) * 0.1
    
    return new_offset

import torch
import torch.nn.functional as F

def update_offset_with_feature_response(offset, features, kernel_size=5, base_factor=0.1, max_factor=0.5):
    """
    使用特征响应动态调整偏移量更新幅度。

    参数:
    offset (torch.Tensor): 输入的偏移量张量 [B, num_offsets, H, W]
    features (torch.Tensor): 网络中的特征图 [B, num_features, H, W]
    kernel_size (int): 领域的尺寸
    base_factor (float): 基础更新因子
    max_factor (float): 最大更新因子

    返回:
    torch.Tensor: 更新后的偏移量张量 [B, num_offsets, H, W]
    """
    batch_size, num_offsets, height, width = offset.size()
    pad = kernel_size // 2

    # 创建均值卷积核
    kernel = torch.ones((num_offsets, num_offsets, kernel_size, kernel_size), device=offset.device) / (kernel_size * kernel_size)

    # 将偏移量进行填充，以处理边界情况
    offset_padded = F.pad(offset, (pad, pad, pad, pad), mode='replicate')

    # 使用卷积计算领域均值
    neighborhood_mean = F.conv2d(offset_padded, kernel, stride=1, padding=0)

    # 计算特征响应
    feature_response = torch.mean(features, dim=1, keepdim=True)
    
    # 计算动态更新因子
    response_factor = torch.sigmoid(feature_response)
    adaptive_factor = base_factor + (max_factor - base_factor) * response_factor

    # 更新偏移量
    new_offset = offset + (offset - neighborhood_mean) * adaptive_factor

    return new_offset
#########################
def update_offset_with_statistical_response(offset, features, kernel_size=5, base_factor=0.1, max_factor=0.5):
    """
    使用特征图的方差、最大值和最小值来动态调整偏移量更新幅度。

    参数:
    offset (torch.Tensor): 输入的偏移量张量 [B, num_offsets, H, W]
    features (torch.Tensor): 网络中的特征图 [B, num_features, H, W]
    kernel_size (int): 领域的尺寸
    base_factor (float): 基础更新因子
    max_factor (float): 最大更新因子

    返回:
    torch.Tensor: 更新后的偏移量张量 [B, num_offsets, H, W]
    """
    batch_size, num_offsets, height, width = offset.size()
    pad = kernel_size // 2

    # 创建均值卷积核
    kernel = torch.ones((num_offsets, num_offsets, kernel_size, kernel_size), device=offset.device) / (kernel_size * kernel_size)

    # 将偏移量进行填充，以处理边界情况
    offset_padded = F.pad(offset, (pad, pad, pad, pad), mode='replicate')

    # 使用卷积计算领域均值
    neighborhood_mean = F.conv2d(offset_padded, kernel, stride=1, padding=0)

    # 计算特征图的方差、最大值和最小值
    feature_mean = torch.mean(features, dim=1, keepdim=True)
    feature_var = torch.var(features, dim=1, keepdim=True)
    feature_max = torch.max(features, dim=1, keepdim=True)[0]
    feature_min = torch.min(features, dim=1, keepdim=True)[0]

    # 归一化特征统计量
    feature_range = feature_max - feature_min + 1e-6  # 避免除以0
    normalized_var = feature_var / feature_range

    # 计算动态更新因子
    response_factor = torch.sigmoid(feature_mean) * (1 + normalized_var)
    adaptive_factor = base_factor + (max_factor - base_factor) * response_factor

    # 更新偏移量
    new_offset = offset + (offset - neighborhood_mean) * adaptive_factor

    return new_offset
########################
def gaussian_kernel(kernel_size, sigma):
    ax = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def update_offset_with_gaussian(offset, features, kernel_size=5, sigma=1.0, base_factor=0.1, max_factor=0.5):
    batch_size, num_offsets, height, width = offset.size()
    pad = kernel_size // 2
    # 创建高斯卷积核
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).to(offset.device)

    kernel = kernel.repeat(num_offsets, num_offsets, 1, 1)  # 修改为 [num_offsets, num_offsets, kernel_size, kernel_size]

    # 将偏移量进行填充
    offset_padded = F.pad(offset, (pad, pad, pad, pad), mode='replicate')
    neighborhood_mean = F.conv2d(offset_padded, kernel, stride=1, padding=0)

    # 计算特征响应
    feature_response = torch.mean(features, dim=1, keepdim=True)

    # 计算动态更新因子
    response_factor = torch.sigmoid(feature_response)
    adaptive_factor = base_factor + (max_factor - base_factor) * response_factor

    # 更新偏移量
    new_offset = offset + F.relu(offset - neighborhood_mean) * adaptive_factor

    return new_offset
#########################
def test_update_offset_with_neighborhood():
    # 设置随机种子以便重现
    torch.manual_seed(0)
    
    # 创建一个随机偏移量张量
    batch_size = 2
    num_offsets = 2
    height = 4
    width = 4
    kernel_size = 3
    
    # 生成一个随机偏移量张量
    offset = torch.randn(batch_size, num_offsets, height, width)
    
    # 打印原始偏移量
    print("Original offset:")
    print(offset)
    
    # 调用更新函数
    updated_offset = update_offset_with_neighborhood(offset, kernel_size)
    
    # 打印更新后的偏移量
    print("\nUpdated offset:")
    print(updated_offset)


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 运行测试
    test_update_offset_with_neighborhood()
    # 创建一个简单的偏移量张量
    # offset = torch.randn(1, 2, 5, 5)
    # print("Original Offset:offset")
    # # 设定目标核大小
    import torch

    # 定义不同的 d_LSKblock 配置
    block_configs = [
        (d_LSKblock3x3, 64),
        (d_LSKblock5x3, 64),
        (d_LSKblock9x7, 64),
        (d_LSKblock7x7, 64),
        (d_LSKblock5x5, 64),
        (d_LSKblock9x9, 64),
    ]

    # 创建输入张量
    input = torch.rand(1, 64, 64, 64).cuda()

    # 循环测试每种配置
    for block_class, channels in block_configs:
        block = block_class(channels).cuda()
        output = block(input)
        print(f"Block: {block_class.__name__}, Input size: {input.size()}, Output size: {output.size()}")
