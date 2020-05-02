# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

from math import sqrt
from functools import partial, lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

"""
权重归一化后的全连接层
"""
def wn_linear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim)) # 权重归一化操作

"""
权重归一化后的全连接层
"""
class WNConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        out = self.conv(input)

        if self.activation is not None:
            out = self.activation(out)

        return out

"""
pad必须是偶数，指定的是该维度的首尾需要pad的行数。指定的维度从 最后一维开始。
这里是使用0将H维的上面pad一行, 相当于将图片向下平移了一行。
"""
def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


"""
这里是使用0将W维的左边pad一行, 相当于将图片向右平移了一行。
"""
def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]

"""
因果卷积
"""
class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]

        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2

            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def forward(self, input):
        out = self.pad(input) # 

        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_() # 将卷积核最后一行后一半置为0, 向上取整。

        out = self.conv(out)

        return out

"""
在最后将channel折半，相互乘后看作, 门阵列需要sigmoid
"""
class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')

        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation(inplace=True)
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)
        out += input

        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T # 返回上三角矩阵, 右上三角， k为对角线+k下元素为0。
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0), # 1 x s x s
        torch.from_numpy(start_mask).unsqueeze(1), # s x 1
    )


class CausalAttention(nn.Module):
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2) # 将每个通道flat为1维。
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2) # 将每个通道flat为1维。
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -1e4) # 在mask为0的地方填0
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value # B*8*(HW)*(C/8)
        out = out.transpose(1, 2).reshape(# B*(HW)*8*(C/8)
            batch, height, width, self.dim_head * self.n_head
        ) # B*H*W*C
        out = out.permute(0, 3, 1, 2)# B*C*H*W

        return out


class PixelBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)

        self.attention = attention

        if attention:
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = torch.cat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)

        else:
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)

        return out


class CondResNet(nn.Module):
    """
    不改变尺寸大小。
    """
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class PixelSNAIL(nn.Module):
    def __init__(
        self,
        shape, # 64 * 64的index矩阵
        n_class, # 512 embedding的word数。
        channel, # 256 中间进行相关卷积时的通道数。
        kernel_size, # 5 * 5的卷积核
        n_block, # 4 pixleblock的个数。
        n_res_block, # 4残差块数量。
        res_channel, # 256
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
        condition = False,
        cond_headpose_gaze=False
    ):
        super().__init__()

        height, width = shape
        self.cond_headpose_gaze = cond_headpose_gaze
        self.condition = condition
        self.shape = shape

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d( # 水平，除了第一行，卷积核的参数置零。
            n_class, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d( # 垂直，除了第一列，卷积核的参数置零。
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (torch.arange(height).float() - height / 2) / height # 坐标归一化，中间为0
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1)) # 高宽坐标作为附加信息。

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            if self.cond_headpose_gaze and not self.condition:
                self.cond_resnet = CondResNet(
                    1, cond_res_channel, cond_res_kernel, n_cond_res_block
                )
            elif not self.cond_headpose_gaze and self.condition:
                self.cond_resnet = CondResNet(
                    n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
                )
            else:
                self.cond_resnet = CondResNet(
                    n_class + 1, cond_res_channel, cond_res_kernel, n_cond_res_block
                )

        if self.cond_headpose_gaze:
            self.expand_cond_pose_gaze = nn.Sequential(
                wn_linear(4, 512),
                nn.ReLU(),
                wn_linear(512, shape[0] * shape[1]),
                                                      )

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, input, condition=None, cond_headpose_gaze=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = (
            F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        )
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical

        background = self.background[:, :, :height, :].expand(batch, 2, height, width) # 1 x 2 x height x width -> 32 x 2 x height x width
        if self.condition or self.cond_headpose_gaze:
            if 'condition' in cache and (self.condition or self.cond_headpose_gaze):
                condition = cache['condition']
                condition = condition[:,:,:height,:]
            else:
                if self.cond_headpose_gaze:
                    cond_h_g = self.expand_cond_pose_gaze(cond_headpose_gaze)
                    cond_h_g = cond_h_g.view(batch, 1, self.shape[0], self.shape[1])

                if self.condition:
                    condition = (
                        F.one_hot(condition, self.n_class)
                        .permute(0, 3, 1, 2)
                        .type_as(self.background)
                    )
                    condition = F.interpolate(condition, scale_factor=2) # condition 发大两倍
                    if self.cond_headpose_gaze:
                        condition = torch.cat([cond_h_g, condition], 1)
                else:
                    condition = cond_h_g
                condition = self.cond_resnet(condition)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]


        for block in self.blocks:
            out = block(out, background, condition=condition)

        out = self.out(out)

        return out, cache
