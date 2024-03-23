# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, dynamical_head= False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points   # 每层关注四个点

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        # 加入 head_embeding++++++++++++++++++++++++++++++++++++++++++++
        # self.head_embeding = nn.Parameter(torch.Tensor(n_heads, d_model//n_heads))   # (8,32)

        self._reset_parameters()
        # 黄
        if dynamical_head == True:
            # self.dynamic_layer = nn.Linear(self.d_model, int(self.d_model/8)*int(self.d_model/8))
            # self.dropout1 = nn.Dropout(0.1)
            # self.dropout2 = nn.Dropout(0.1)
            self.gates_layer = nn.Linear(self.d_model, 2*self.n_heads)
        self.attention_position = []
        self.attention_weight = []

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # ++++++++++黄  换成别的初始化
        # xavier_uniform_(self.sampling_offsets.weight.data, mode='fan_in')
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        # nn.init.normal_(self.head_embeding)   #  head_embeding+初始化+++++++++++

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, \
        input_level_start_index, input_padding_mask=None, save_map= False, cls_o = None, dynamic_head =False):    # save attention_map ++++map
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape              # 被查询的 shape [1,N,256]
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value.masked_fill_(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++DDD
        if dynamic_head == True:
            # parameters = self.dynamic_layer(query).view(N, Len_q, self.d_model//8, self.d_model//8)   # b,1000,36*36
            gate,gate_cls = self.gates_layer(query).unsqueeze(-1).split(self.n_heads, 2)   # b,1000,36*36
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++DDD
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2) # 通过全连接得到偏移量[1,1000,8,4,4，2]
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)  # 通过全连接得到注意力权重  []
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points) # [1,1000,8,4,4] 8个heads 4levels 每层4个点
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:  # [1,1000,4,4]               # 取出anchor 的中心坐标   [1,1000,8,4,4,2]
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5  # sampling_offsets / self.n_points相对于一半高宽的偏移量，关注点都在anchor范围内
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++DDD
        if dynamic_head == True:
            # output01 = torch.matmul(output.view(N, Len_q, self.n_heads,-1), parameters).flatten(-2)
            # output = output+self.dropout1(output01)
            output =(output.view(N, Len_q, self.n_heads,-1)*gate).flatten(-2)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++DDD
        # 加入 head_encoding+++++++++++++++++++++++++++++++++++++++
        # output = output + self.head_embeding.flatten().unsqueeze(0).unsqueeze(0).repeat(N, Len_q,1)   # head embeding (8,32)
        # 加入 head_encoding+++++++++++++++++++++++++++++++++++++++
        output = self.output_proj(output)                   # output 1,20906,256
        # +++++++++++++++++黄
        if save_map ==True:
            self.attention_position = sampling_locations
            self.attention_weight = attention_weights
        # +++++++++++++++++黄 # [1,1000,8,4,4]
        if bool(cls_o) == True:
            # cls_ = 8-cls_o
            # zero_fill = torch.zeros_like(sampling_locations[:, :, cls_o:8, ...])
            # sampling_locations_cls = torch.cat((zero_fill,sampling_locations[:, :, cls_:8, ...]),2)
            # zero_fill = torch.zeros_like(attention_weights[:, :, cls_o:8,...])
            # sampling_weights_cls = torch.cat((zero_fill,attention_weights[:, :, cls_:8,...]), 2)
            # cls_out = MSDeformAttnFunction.apply(
            #     value, input_spatial_shapes, input_level_start_index, sampling_locations_cls, sampling_weights_cls,
            #     self.im2col_step)
            if dynamic_head == True:
                # cls_out01 = torch.matmul(cls_out.view(N, Len_q, self.n_heads, -1), parameters).flatten(-2)
                # cls_out = cls_out+self.dropout2(cls_out01)
                cls_out = (output.view(N, Len_q, self.n_heads, -1) * gate_cls).flatten(-2)
            cls_out = self.output_proj(cls_out)
            return output, cls_out
        #++++++++++++++++++黄
        return output   # ++++++++++++++++++++黄 # return sampling_locations [1,1000,8,4,4,2](绝对位置) 和  attention_weights   [1,1000,8,4,4]
