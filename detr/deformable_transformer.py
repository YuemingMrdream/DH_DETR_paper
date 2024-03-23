# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy

from torch._C import device, dtype
from torch.random import seed
from .DehomoCodingGenerator import DehomoCodingGenerator
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from config import config
from misc import inverse_sigmoid
from ops.modules import MSDeformAttn
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .ra import RectifiedAttn

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 rectified_attention=0, dense_query=0, aps=0, AMself=0, DeA=1,ARelation=1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer,
                                                    num_decoder_layers, return_intermediate_dec,
                                                    d_model, rectified_attention, dense_query, aps, AMself, DeA, ARelation, two_stage_num_proposals)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))



        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 4)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn) or isinstance(m, RectifiedAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # 可视化encode的初始框的函数
    def visual_init_reference_out(self,enc_outputs_class, init_reference_out):
        # torch.save({'dtboxes':init_reference_out}, r'visual/cur_init_ref.pt')
        import demo
        topk = 1000
        cur_ = torch.cat([torch.topk(enc_outputs_class.sigmoid()[..., 0], topk, dim=1)[0].unsqueeze(-1), init_reference_out], dim=-1)
        cur_= cur_.cpu().numpy()
        cur_dt = []
        for ob in cur_[0]:
            cur_dt.append({'box':ob[1:], 'score':ob[0]})
        demo.paint_images_heatmap({'dtboxes':cur_dt})

        return None

    def distinct_encoding_mask(self, topk_enc_outputs_class):
        # hack implementation for iterative bounding box refinement
        class_logits = topk_enc_outputs_class

        prev_logits = class_logits.detach()
        scores = prev_logits.max(-1).values.sigmoid()  # (1,1000)
        # 非对称mask++++++++++++++++++++++++++++++++++++
        scores_x = scores.unsqueeze(-1).repeat_interleave(self.two_stage_num_proposals, -1).unsqueeze(-1)  # (1,1000,.1000.,1)
        scores_y = scores.unsqueeze(-1).transpose(-2, -1).repeat_interleave(self.two_stage_num_proposals, -2).unsqueeze(-1)  # (1,1,1000,1)  # (1,.1000.,1000,1)
        # 取x<y的值
        asymmetric_mask = scores_x < scores_y
        # 置信度小于0.1的不关注
        low_mask = scores_y > config.asymmetrical_low
        high_score_mask = asymmetric_mask * low_mask
        # +++++++++++++++++++++++++++++++++++++++++++++
        seed_mask = scores > config.score_thr  # seed_mask 是预测置信度大于某个阈值的 一般假正率比较小
        cur_all_mask = (~seed_mask) & (scores > config.floor_thr)  # ??????
        return  cur_all_mask, seed_mask, high_score_mask, prev_logits# return  "mask": cur_all_mask, "seed_mask": seed_mask,"high_score_mask": high_score_mask

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):  # 好像是四个特征图
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)                    # 每一层特征图的position_embedding [1,8976,256]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # 加入层间位置编码
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # lvl_pos_embed_flatten = [1,11935,256]
        # src 来自于backbone,src= [1,11935,256](第二维是输入的特征数量)
        # spatial_shapes 是输入的四层特征图的大小[[102,88],[51,44],[26,22],[13,11]]
        # level_start_index 是每层开始的索引(0,9876,11220,11792)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # lvl_pos_embed_flatten  [1,11935,256] 是结合层编码的位置编码，
        # prepare input for decoder
        bs, _, c = memory.shape

        #  false ,好像如果是两阶段的话是通过encoder的注意力图 来生成decoder的proposals
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR，查看通过encoder初始化的decoder_queries
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            enc_outputs_coord_unact = self.decoder.bbox_embed[-1](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            # init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

            topk_enc_outputs_class = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[0].unsqueeze(-1)
            cur_all_mask, seed_mask, high_score_mask, prev_logits = self.distinct_encoding_mask( topk_enc_outputs_class)

        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)     # 原来的query是512，现在切成了256和256
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()# 是query_embed预测的初始anchor坐标（绝对坐标），是显示anchor出来之前的权宜之计
            # init_reference_out = reference_points

        # 可视化
        # self.visual_init_reference_out(enc_outputs_class,init_reference_out)

        # decoder   memory来自于encoder  reference_points 是当前query_embed预测的box （共用参数）,query_embed 是query的可学习位置编码向量
        box, cls, mask = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,cur_all_mask, seed_mask, high_score_mask,prev_logits)

        # 可视化
        # self.visual_init_reference_out(cls[5],box[5])
        # 测试enc输出
        # topk_enc_outputs_class = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[0].unsqueeze(-1)

        if self.two_stage:
            return box, cls, mask, enc_outputs_class, enc_outputs_coord_unact
            # 测试enc输出
            # return box, cls, mask, topk_enc_outputs_class, topk_coords_unact
        return box, cls, mask, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):                                                  #前向网络
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))   #全连接 -relu- dropout-全连接
        src = src + self.dropout3(src2)                                          #残差连接
        src = self.norm2(src)                                                    #layer normalization
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)  #特征图编码间的自注意力机制
        src = src + self.dropout1(src2)                                                                                               #残差连接
        src = self.norm1(src)                                                                                                         #layer normalization
        # ffn
        src = self.forward_ffn(src)                                                                                                   #前向网络
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]   # valid_ratios是什么
        return reference_points
    # spatial_shapes 是输入的四层特征图的大小[[102,88],[51,44],[26,22],[13,11]]  src= [1,11935,256](第二维是输入的特征数量)  level_start_index 是每层开始的索引(0,9876,11220,11792)
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)  # 假设原图为 [0-1][0-1]  映射每个点的坐标到原图  referenc_points = [1,11935,4,2] 是像素点中心的位置  (倒数第二个维度是重复的，由valid_ratios决定，但是默认1)
        for _, layer in enumerate(self.layers):   # pos 是 output对应的位置编码  referenc_points = [1,11935,4,2]
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention   改进的deformable attention   n_points 为4 即4*4
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, dynamical_head= False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Delete self attention among queries
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # # self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = None
        # self.norm3 = nn.LayerNorm(d_model)

        # APS
        self.aps = None

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, attention_position, attention_weight,
                src_padding_mask=None, attn_mask=None, seed_mask=None,cur_all_mask=1, pred_boxes=None,last_second = 0,high_score_mask =None,cls_o = True, prev_scores=None
                ):
        if self.aps is not None:
            # tgt.detach_()
            # src.detach_()
            tgt, attn_mask = self.aps(tgt, seed_mask.float().unsqueeze(-1), pred_boxes, attention_position, attention_weight, high_score_mask,prev_scores=prev_scores)
            cur_all_mask = cur_all_mask.float().unsqueeze(-1)

            tgt = self.norm2(tgt * cur_all_mask)  # 只有低置信度的更新

        # q = k = self.with_pos_embed(tgt, query_pos)    # 这个query_pos 并没有更新！！！！在DAB中才有好的效果
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # tgt = tgt + self.dropout2(tgt2)    # tgt2 密集，因为v还经过全连接 ，tgt稀疏
        # tgt = self.norm3(tgt)  # 只有低置信度的更新

        if cls_o !=0:
            tgt2, cls_out= self.cross_attn(self.with_pos_embed(tgt, query_pos) * cur_all_mask, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask,save_map = False, cls_o= cls_o,dynamic_head =False)
        else:
            tgt2= self.cross_attn(self.with_pos_embed(tgt, query_pos) * cur_all_mask, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask,save_map = False, cls_o= cls_o,dynamic_head =False)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt * cur_all_mask)
        # ffn
        tgt = self.forward_ffn(tgt)
        cls_output = tgt

        attention_position = None
        attention_weight = None
        return tgt, attention_position, attention_weight, cls_output


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, d_model=256, rectified_attention=0, dense_query=0, aps=0, AMself=0, DeA=1 , ARelation=1, num_query = 1000):
        super().__init__()
        _layers = [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        self.aps = aps
        self.Amself = AMself
        self.DeA = DeA
        self.ARelation = ARelation
        self.num_query = num_query
        for i in range(num_layers - aps, num_layers):
            _layers[i].aps = DehomoCodingGenerator(d_model, Amself =AMself, num_query= num_query)
            _layers[i].norm2 = nn.LayerNorm(d_model)   # 加入aps之后的norm

        for i in range(num_layers - rectified_attention, num_layers):
            _layers[i].cross_attn = RectifiedAttn(d_model)       # rectifiedAttention 是用栅格初始化注意点
        self.layers = nn.ModuleList(_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.dense_query = dense_query

    def forward(self, tgt: Tensor, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, cur_all_mask=1, seed_mask=None, high_score_mask=None,prev_logits= None):   # src [1, 11935,256]   是来自encoder的特征
        # reference_points   是当前query_embed预测的box （共用参数）
        output = tgt
        # class_emb
        cls_out = tgt
        bs, num_queries = tgt.shape[:2]
        scores= None  #+++++++++黄
        intermediate_pred_boxes = []
        intermediate_class_logits = []
        intermediate_masks = []
        seed_mask = None
        cur_all_mask = 1
        prev_logits = None
        pred_boxes = reference_points

        attention_position = 0
        attention_weight = 0
        high_score_mask = None
        last_second = False
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            with torch.no_grad():
                attn_mask = None
                if lid >= self.num_layers - self.dense_query:
                    tgt_masks = []
                    for pred in reference_points:
                        tgt_masks_ = torch.ones((num_queries, num_queries), dtype=bool, device=src.device)
                        boxes = box_cxcywh_to_xyxy(pred)
                        score = 1 - generalized_box_iou(boxes, boxes)
                        top_idx = torch.argsort(score, dim=-1)[:, :100] # returns a longtensor
                        tgt_masks_.scatter_(1, top_idx, 0)
                        tgt_masks.append(tgt_masks_)
                    attn_mask = torch.stack(tgt_masks, dim=0).repeat_interleave(8, 0)


            output, attention_position, attention_weight, cls_output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                                                                 src_level_start_index,src_padding_mask= src_padding_mask , attention_position = attention_position, attention_weight = attention_weight ,attn_mask=attn_mask, seed_mask=seed_mask,
                                                                 cur_all_mask=cur_all_mask, pred_boxes=reference_points,last_second =last_second, high_score_mask =high_score_mask , cls_o =self.DeA,prev_scores=scores)

            class_logits = self.class_embed[lid](cls_output)
            if layer.aps is None:
                pred_boxes = self.bbox_embed[lid](output) + inverse_sigmoid(reference_points)
                pred_boxes = pred_boxes.sigmoid()

            if layer.aps is not None:
                class_logits[~cur_all_mask] = prev_logits[~cur_all_mask]
                class_logits[~cur_all_mask & ~seed_mask] = float('-inf')
                intermediate_masks.append({"mask": cur_all_mask, "seed_mask": seed_mask})  # 加入 cascade_mask
            else:
                intermediate_masks.append(None)

            reference_points = pred_boxes.detach()  # ---推测的框坐标        reference_points 每层会更新的
            prev_logits = class_logits.detach()
            if lid >= self.num_layers - self.aps - 1:
                scores = prev_logits.max(-1).values.sigmoid() # (1,1000)
                if self.ARelation == True:
                    # # 非对称mask++++++++++++++++++++++++++++++++based on confidence
                    scores_x = scores.unsqueeze(-1).repeat_interleave(self.num_query, -1).unsqueeze(-1)                      # (1,1000,.1000.,1)
                    scores_y = scores.unsqueeze(-1).transpose(-2,-1).repeat_interleave(self.num_query, -2).unsqueeze(-1)     # (1,1,1000,1)  # (1,.1000.,1000,1)
                    # 取x<y的值
                    asymmetric_mask = scores_x < scores_y
                    # 置信度小于0.1的不关注
                    low_mask = scores_y > config.asymmetrical_low
                    high_score_mask = asymmetric_mask * low_mask
                    #+++++++++++++++++++++++++++++++++++++++++++++
                seed_mask = scores > config.score_thr                       # seed_mask 是预测置信度大于某个阈值的 一般假正率比较小
                cur_all_mask = (~seed_mask) & (scores > config.floor_thr)   # 置信度小于0.7 大于0.05的为1

            if self.return_intermediate:
                intermediate_pred_boxes.append(pred_boxes)
                intermediate_class_logits.append(class_logits)

        if self.return_intermediate:
            return intermediate_pred_boxes, intermediate_class_logits, intermediate_masks

        return reference_points, class_logits, {"mask": cur_all_mask, "seed_mask": seed_mask}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        dense_query=args.dense_query,
        rectified_attention=args.rectified_attention,
        aps=args.aps,
        AMself=args.AMself,
        DeA=args.DeA,
        ARelation=args.ARelation
        )


