import os,sys
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from config import config
from misc import inverse_sigmoid
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
import torch, pdb

class DehomoCodingGenerator(nn.Module):

    def __init__(self, d_model, activation = 'relu', neighbors = 10, Amself= 0,num_query=1000):
        
        super().__init__()

        assert neighbors > 0
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.fc3 = nn.Linear(2*d_model + 64, d_model)

        self.linear5 = nn.Linear(d_model, d_model)

        # Amself
        self.Amself = Amself
        self.num_query = num_query

        if self.Amself:
            self.linear7 = nn.Linear(272, d_model)   # d_model 256
            self.linear8 = nn.Linear(d_model, d_model)
            self.linear9 = nn.Linear(2 * d_model, d_model)
        else:
            self.linear3 = nn.Linear(d_model, d_model)
            self.linear4 = nn.Linear(d_model, d_model)
        self.activation = _get_activation_fn(activation)
        self.top_k = neighbors



    def _recover_boxes(self, container):

        assert np.logical_and('reference_points' in container, 'pred_boxes' in container)

        reference_points = container['reference_points']
        pred_boxes = container['pred_boxes']
        
        if reference_points.shape[-1] == 4:
            new_reference_points = pred_boxes + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        else:
            assert reference_points.shape[-1] == 2
            new_reference_points[..., :2] = pred_boxes[..., :2] + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        
        tmp = new_reference_points.detach()
        
        return tmp

    @torch.no_grad()
    def sin_cos_encoder(self, boxes, indices):
        # 这一部分的主要工作是对detected boxes之间的关系进行sin/cos编码.   # 相对位置
        eps = 1e-7
        cur_boxes = boxes.unsqueeze(2)
        neighbors = torch.gather(boxes.unsqueeze(1).repeat_interleave(self.num_query, 1), 2, indices[..., :4])

        delta_ctrs = neighbors - cur_boxes
        position_mat = torch.log(torch.clamp(torch.abs(delta_ctrs), eps))    # 把高宽等长度log了因为长款都是小数，所以log是之后 position_mat是负数
        return self._extract_position_embedding(position_mat)

    @torch.no_grad()
    def score_sin_cos_encoder(self,scores, indices):
        eps = 1e-7
        cur_scores = scores.unsqueeze(-1).unsqueeze(2)
        neighbors = torch.gather(scores.unsqueeze(-1).unsqueeze(1).repeat_interleave(self.num_query, 1), 2, indices[..., :1])

        # delta_ctrs = neighbors - cur_scores
        position_mat = torch.log(torch.clamp(torch.abs(neighbors), eps))    # 把高宽等长度log了因为长款都是小数，所以log是之后 position_mat是负数
        return self._extract_position_embedding(position_mat)




    def _extract_position_embedding(self, position_mat, num_pos_feats=64,temperature=1000):

        num_pos_feats = 128
        temperature = 10000
        scale = 2 * np.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device = position_mat.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # scale * position_mat.unsqueeze(-1) 维度[1,1000,10,4,1]   pos_mat [1,1000,10,4,128]
        pos_mat = scale * position_mat.unsqueeze(-1) / dim_t.reshape(1, 1, 1, 1, -1)
        pos = torch.stack((pos_mat[:, :, :,:, 0::2].sin(), pos_mat[:, :, :,:, 1::2].cos()), dim=4).flatten(3) #应该是最后一个维度相加
        return pos   # [1,1000,10,512]

    def _extract_attention_position_embedding(self, position_mat, num_pos_feats=64, temperature=1000):

        num_pos_feats = 128
        temperature = 10000
        scale = 2 * np.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=position_mat.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # scale * position_mat.unsqueeze(-1) 维度[1,1000,10,4,1]   pos_mat [1,1000,10,4,128]
        pos_mat = scale * position_mat.unsqueeze(-1) / dim_t.reshape(1, 1, 1, 1, -1)
        pos = torch.stack((pos_mat[:, :, :,:, 0::2].sin(), pos_mat[:, :, :,:, 1::2].cos()), dim=4).flatten(3)  # 应该是最后一个维度相加
        # pos = torch.stack((pos_mat[:, :, :, :,0::2].sin(), pos_mat[:, :, :, :, 1::2].cos()), dim=4).flatten(3)
        # 把position_mat 的四个维度位置编码后128 再cat起来变成512维
        return pos  # [1,1000,10,512]

    def attention_map_encoder(self, boxes, indices, attention_position, attention_weight):
        # 这一部分的主要工作是对detected boxes之间的关系进行sin/cos编码.   # 相对位置
        eps = 1e-7
        cur_boxes = boxes.unsqueeze(2)
        # 选权重最大的8个点

        # 加入neighbors    [1,1000,10,8,4,4,2]     indice=1????
        # neighbors_weight = torch.gather(attention_weight.unsqueeze(1).repeat_interleave(1000, 1), 2, indices[..., :1,None,None].repeat_interleave(8,dim = -3).repeat_interleave(4,dim = -2).repeat_interleave(4,dim = -1))
        neighbors_position = torch.gather(attention_position.unsqueeze(1).repeat_interleave(1000, 1), 2, indices[..., :1,None,None,None].repeat_interleave(8,dim = -4).repeat_interleave(4,dim = -3).repeat_interleave(4,dim = -2).repeat_interleave(2,dim = -1))

        # 16个点对应相减，变成16*16+++++++拓展转置
        neighbors_position = neighbors_position.flatten(-3, -2).unsqueeze(-2).repeat_interleave(16, -2).transpose(-2, -3)   # [1,1000,10,8,16,16,2]
        attention_position = attention_position.unsqueeze(2).repeat_interleave(10, 2).flatten(-3, -2).unsqueeze(-2).repeat_interleave(16, -2) # [1,1000,1,8,4,4,2] ----[1,1000,10,8,16,16,2]
        delta_ctrs = neighbors_position - attention_position  # [1,1000,10,8,16,16,2]

        self_delta_ctrs = attention_position.transpose(-2, -3) - attention_position
        # 矩阵消重，
        tri_mask = torch.triu(torch.ones(16,16), diagonal=0).flatten(0).to(delta_ctrs.device)
        # tensor([[1., 1., 1., 1., 1.],
        #         [0., 1., 1., 1., 1.],
        #         [0., 0., 1., 1., 1.],
        #         [0., 0., 0., 1., 1.],
        #         [0., 0., 0., 0., 1.]])

        tri_mask = torch.where(tri_mask > 0)  # 返回上三角的索引
        delta_ctrs = torch.index_select(delta_ctrs.flatten(-3,-2), dim=-2, index=tri_mask[0])   # [1,1000,10,8,16,16,2] -[1,1000,10,8,256,2]-[1,1000,10,8,上三角136,2]
        self_delta_ctrs = torch.index_select(self_delta_ctrs.flatten(-3,-2), dim=-2, index=tri_mask[0])   # [1,1000,10,8,16,16,2] -[1,1000,10,8,256,2]-[1,1000,10,8,上三角136,2]


        # 按照 dx+dy 排序,加快收敛
        _, idx_ = torch.sort(torch.abs(delta_ctrs).sum(-1), -1, descending=False)  # [1, 1000, 10, 8, (16+1)*16/2,2]
        sorted_delta_ctrs = torch.gather(delta_ctrs, -2, idx_.unsqueeze(-1).repeat_interleave(2, -1))
        _, idx_ = torch.sort(torch.abs(self_delta_ctrs).sum(-1), -1, descending=False) # [1, 1000, 10, 8, (16+1)*16/2,2]
        sorted_self_delta_ctrs = torch.gather(self_delta_ctrs, -2, idx_.unsqueeze(-1).repeat_interleave(2, -1))

        # 下面是公用权重
        inter_relation = (self.linear8((self.activation(self.linear7(sorted_delta_ctrs.flatten(-2)))))).mean(-2)# [1,1000,10,8,(16+1)*16]--[1,1000,10,8,256]-[1,1000,10,256]
        self_relation = (self.linear8((self.activation(self.linear7(sorted_self_delta_ctrs[:, :, 0, ...].flatten(-2)))))).unsqueeze(2).mean(-2) # [1,1000,1,8,(16+1)*16]--[1,1000,1,8,256]-[1,1000,1,256]

        attention_relation = self.linear9(torch.cat((self_relation.repeat_interleave(10, -2), inter_relation), dim=-1)) # [1,1000,10,256]--cat---[1,1000,10,512]-linear---[1,1000,10,256]

        return attention_relation  # [1,1000,10,256]

    def forward(self, tgt, seed_mask, pred_boxes, attention_position, attention_weight, high_score_mask,prev_scores): # +++++黄
        bs, num_queries = pred_boxes.shape[:2]
        
        # recover the predicted boxes
        box_xyxy = box_cxcywh_to_xyxy(pred_boxes)

        # compute the overlaps between boxes and reference boxes
        ious = torch.stack([box_iou(boxes, boxes)[0] for boxes in box_xyxy]) # [1,1000,1000]

        attn_mask = ious >= config.iou_thr  #[1,1000,1000]

        # use masking to mask the overlap，seed_mask 是什么
        neg_mask = (1 - seed_mask)
        # overlaps = ious * seed_mask.permute(0, 2, 1) * neg_mask
        if high_score_mask!=None:
            overlaps = ious *high_score_mask.squeeze(-1) * neg_mask    # 改 4.1

            c = tgt.shape[-1]
            indices = torch.argsort(-overlaps)[..., :self.top_k].unsqueeze(-1).repeat_interleave(c, dim = -1) # [1,1000,10,256]

            nmk = torch.gather(high_score_mask.squeeze(-1), 2, indices[..., 0])  # mask 后面999，998
            ious = torch.gather(overlaps, 2 ,indices[..., 0])
        else:
            overlaps = ious * seed_mask.permute(0, 2, 1) * neg_mask

            c = tgt.shape[-1]
            indices = torch.argsort(-overlaps)[..., :self.top_k].unsqueeze(-1).repeat_interleave(c, dim=-1)

            nmask = seed_mask.permute(0, 2, 1).repeat_interleave(num_queries, dim=1)
            nmk = torch.gather(nmask, 2, indices[..., 0])
            ious = torch.gather(overlaps, 2, indices[..., 0])
        mk = nmk * (ious >= config.iou_thr)
        overs = (ious * mk).unsqueeze(-1).repeat_interleave(64, dim = -1)     # [BN,1000,10,64] ious信息(copy)


        if self.Amself:
            # query的attention map编码
            waves_features = self.attention_map_encoder(pred_boxes, indices, attention_position, attention_weight)
            cur = self.linear2(self.activation(self.linear1(tgt)))  # original decoder embedding
            # mask the features.
            cur_tgt = cur * neg_mask + (waves_features * mk.unsqueeze(-1)).max(dim = 2)[0]
        else:
            #adding dehomo coding
            id_token = self.linear2(self.activation(self.linear1(tgt)))
            id_token = self.norm2(id_token)
            neighbors_id = torch.gather(id_token.unsqueeze(1).repeat_interleave(self.num_query, 1), 2, indices[..., :256])
            id_feature = id_token.unsqueeze(2) - neighbors_id                  # Subtract
            features = id_feature
            features = self.linear4(self.activation(self.linear3(features)))   # surrounding iou
            cur_tgt =(features * mk.unsqueeze(-1)).max(dim = 2)[0]             # asymmetrical

        # update feature of target
        if self.Amself:
            cur_tgt = self.activation(self.linear5(cur_tgt))* neg_mask  # 高置信度的tgt任然有值
        else:
            cur_tgt = tgt + self.activation(self.linear5(cur_tgt)) * neg_mask    # 高置信度的tgt全为0

        return cur_tgt, attn_mask

def _get_activation_fn(activation):

    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_relation(args):

    return DehomoCodingGenerator(d_model = args.hidden_dim, activation ='relu', neighbors = 10)

def build_DehomoCodingGenerator(d_model=256, activation='relu', neighbors = 10):

    return DehomoCodingGenerator(d_model, activation, neighbors)
