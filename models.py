# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
from math import sqrt
import torch.nn as nn
import torch.utils.data 
import math
import os
import torch.nn.functional as F
from torchvision.models import vgg16

class gazeNet_single(nn.Module):
    def __init__(self, pretrained_path='/disks/disk0/fjl/vgg16-397923af.pth'):
        super(gazeNet_single, self).__init__()
        VGG = vgg16()
        if os.path.isfile(pretrained_path):
            VGG.load_state_dict(torch.load(pretrained_path))
            print("load pretrained weight successed")
        self.feature = VGG.features
        self.feature._modules['4'] = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.feature._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature._modules['9'] = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.fc_1 = nn.Sequential(
            nn.Linear(4 * 7 * 512, 1000),
            nn.ReLU(inplace=True)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(1002, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2)
        )

    def forward(self, img, head_pose):
        x = self.feature(img)
        feature = x.view(x.size()[0], -1)
        fc_1 = self.fc_1(feature)
        fc_1_h = torch.cat([fc_1, head_pose], 1)
        fc_2 = self.fc_2(fc_1_h)
        return  fc_2

class single(nn.Module):
    def __init__(self):
        super(single, self).__init__()
        self.conv1 = nn.Sequential(
                         nn.Conv2d(1,20,kernel_size=5,stride=1,padding=0,dilation=1),
                         nn.ReLU(),
                         # nn.BatchNorm2d(20),
                    )
                         
        self.conv2 = nn.Sequential(
                         nn.MaxPool2d(2,stride=2),
                         nn.Conv2d(20,50,kernel_size=5,stride=1,padding=0,dilation=1),
                         nn.ReLU(),
                         # nn.BatchNorm2d(20),
                         nn.MaxPool2d(2,stride=2))
        self.fc = nn.Sequential(
            nn.Linear(50*12*6, 500),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(502, 2)
            )
        
    def forward(self,x, head_pose):
        x = self.conv2(self.conv1(x))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = torch.cat([x, head_pose], 1)
        x = self.fc1(x)
        return  x

class BaseMode(nn.Module):
    def __init__(self):
        super(BaseMode,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.ReLU()
                         )
                         
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2,stride=2))

        self.conv3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.ReLU()
                                  )

        self.conv4 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2, stride=2))

        self.conv5 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.ReLU()
                                  )
        self.conv6 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0,dilation=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2, stride=2))
        
    def forward(self,x):
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        x = x.view(x.size()[0], -1)
        return  x


class E_Net(nn.Module):
    def __init__(self):
        super(E_Net, self).__init__()
        self.probab_mode_l = BaseMode()
        self.probab_mode_r = BaseMode()
        self.fc5_l = nn.Sequential(
            nn.Linear(256*1*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())
        self.fc5_r = nn.Sequential(
            nn.Linear(256*1*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())
        self.fc6 = nn.Sequential(
            nn.Linear(1000,2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img_l,img_r):
        imge_l = self.probab_mode_l(img_l)
        imge_r = self.probab_mode_r(img_r)

        img_pro_l = self.fc5_l(imge_l)
        img_pro_r = self.fc5_r(imge_r)
        pro_l_r = self.fc6(torch.cat([img_pro_l,img_pro_r],1))
        pro = self.softmax(pro_l_r)
        return pro

class AR_Net(nn.Module):
    def __init__(self):
        super(AR_Net, self).__init__()
        self.AR_up = AR_Net_up()
        self.AR_down = AR_Net_down()
        self.fc4 = nn.Sequential(
            nn.Linear(1504,6))


    def forward(self, img_l, img_r ,head_pose_l,head_pose_r):
        imge_up = self.AR_up(img_l,img_r)
        imge_down = self.AR_down(img_l,img_r)
        result = self.fc4(torch.cat([imge_up,imge_down,head_pose_l,head_pose_r],1))
        left_gaze = result[:, 0:3 ]
        right_gaze = result[:, 3:]
        return  left_gaze, right_gaze


class AR_Net_down(nn.Module):
    def __init__(self):
        super(AR_Net_down, self).__init__()
        self.eyeModel_l = BaseMode()
        self.eyeModel_r = BaseMode()

        self.fc2_l = nn.Sequential(
            nn.Linear(256 * 1 * 4, 500),
            nn.ReLU())
        self.fc2_r = nn.Sequential(
            nn.Linear(256 * 1 * 4, 500),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU())

    def forward(self, l,r):
        image_l = self.eyeModel_l(l)
        image_r = self.eyeModel_r(r)

        fc2_l = self.fc2_l(image_l)
        fc2_r = self.fc2_r(image_r)

        fc2_l_r = torch.cat([fc2_l, fc2_r], 1)
        fc3_l_r = self.fc3(fc2_l_r)
        return  fc3_l_r                              ##output 500

class AR_Net_up(nn.Module):
    def __init__(self):
        super(AR_Net_up, self).__init__()
        self.eyeModel_l = BaseMode()                           # not sharing the weight
        self.eyeModel_r = BaseMode()

        self.fc1_l = nn.Sequential(
            nn.Linear(256*1*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())
        self.fc1_r = nn.Sequential(
            nn.Linear(256*1*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU())

    def forward(self, l,r):
        imag_l = self.eyeModel_l(l)
        imag_r = self.eyeModel_r(r)

        fc1_l = self.fc1_l(imag_l)
        fc1_r = self.fc1_r(imag_r)

        eye_up = torch.cat((fc1_l,fc1_r),1)
        return  eye_up                       #output 1000

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), # 按深度卷积
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV2_modified(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2_modified, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        # x = self.classifier(x)
        return x # 1280

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MBGazeNet(nn.Module):
    def __init__(self, width_mult=1., fcs = [ 512, 256, 2]):
        super(MBGazeNet, self).__init__()
        self.single_l = MobileNetV2_modified(width_mult)
        self.fcs_l = nn.Sequential(
            nn.Linear(make_divisible(1280 * width_mult), fcs[0]))
        self.fcs_l_1 = nn.Sequential(
            nn.Linear(fcs[0] + 2, fcs[1]),
            nn.Dropout(0.1),
            nn.Linear(fcs[1], fcs[2])
        )

    def forward(self, left_img, left_headpose):
        left_feature = self.single_l(left_img)
        left_feature = self.fcs_l(left_feature)
        left_feature = torch.cat([left_feature, left_headpose], 1)
        left_gaze = self.fcs_l_1(left_feature)
        return left_gaze

class filter_SEblock(nn.Module):
    def __init__(self, in_dims, out_dims, stride=1):
        super(filter_SEblock, self).__init__()
        self.preact = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dims)
        )
        self.W = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_dims, out_dims // 16, 1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims // 16, in_dims, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, left_feature, right_feature):
        x = torch.cat([left_feature, right_feature], 1)
        pre = self.preact(x)
        w = self.W(pre)
        # a, b = torch.split(w, [256, 256], 1)
        # a = torch.squeeze(a)
        # b = torch.squeeze(b)
        return w * x

class filter_identity(nn.Module):
    def __init__(self, in_dims, out_dims, stride=1):
        super(filter_identity, self).__init__()

    def forward(self, left_feature, right_feature):
        x = torch.cat([left_feature, right_feature], 1)
        return x

class filter_identity_inform(nn.Module):
    def __init__(self, query_channel, key_channel):
        super(filter_identity_inform, self).__init__()

    def forward(self, query, key):
        return torch.cat([query, key], 1)


class filter_ConvAttention(nn.Module):
    def __init__(self, in_dims, out_dims, n_head=8, dropout=0.1):
        super(filter_ConvAttention, self).__init__()
        
        self.query = nn.Conv2d(in_dims, out_dims * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.key = nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.value = nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
        self.query_WN = wn_linear(out_dims * 2, in_dims)
        self.key_WN = wn_linear(out_dims, in_dims)
        self.value_WN = wn_linear(out_dims, in_dims)
        self.dim_head = in_dims // n_head
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, left_feature, right_feature):
        combine_f = torch.cat([left_feature, right_feature], 1)
        batch, _, height, width = combine_f.shape
        query = self.query(combine_f)
        key = self.key(combine_f)
        value = self.value(combine_f)

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        value_flat = value.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query_WN(query_flat))
        key = reshape(self.key_WN(key_flat)).transpose(2, 3)
        value = reshape(self.value_WN(value_flat))

        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        attn = torch.softmax(attn, 3)
        attn = self.dropout(attn)
        out = attn @ value
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)
        
        return out.contiguous()

class filter_FcAttention(nn.Module):
    def __init__(self, query_channel, key_channel, n_head = 2, d_k = 2, d_v = 1, dropout=0.3):
        super(filter_FcAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.query = wn_linear(query_channel, n_head * d_k)
        self.key = wn_linear(key_channel, n_head * d_k)
        self.value = wn_linear(key_channel, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, key_channel)

        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, query, key):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), key.size(1)

        residual = query
        q = self.query(query)# .view(batch, len_q, n_head, d_k)
        q = q.view(batch, len_q, n_head, d_k)
        k = self.key(key).view(batch, len_k, n_head, d_k)
        v = self.value(key).view(batch, len_k, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) / sqrt(32)
        attn = self.dropout(torch.softmax(attn, 3))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, len_v, -1)
        out = self.dropout_1(self.fc(out))
        return torch.cat([residual, out], 1)


def wn_linear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim)) # 权重归一化操作
