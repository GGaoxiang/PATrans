import torch.nn.functional as F
from torch import nn
import torch
import os
import time
from patrans_core.PATrans_module.layers.basic import DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d
from patrans_core.PATrans_module.layers.attention import GlobalPatternAware,  LocalPatternAware, MultiheadAttention
from patrans_core.PATrans_module.layers.position import PositionEmbeddingSine
import random

def _get_norm(indim, type='ln', groups=8):
    if type == 'gn':
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")


class GlobalLocalPATrans(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 num_head=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 global_dropout=0.,
                 local_dropout=0.,
                 patch_num=4,
                 max_dis=7,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 cross_att_pos = False,
                 final_norm=True):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                GlobalLocalPATransBlock(d_model, num_head,
                                        dim_feedforward, droppath_rate,
                                        global_dropout, local_dropout, patch_num,
                                        droppath_lst, max_dis, activation))
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model, type='ln') for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

    def forward(self,
                tgt,
                ref_embs,
                curr_id_emb=None,
                self_pos=None,
                pos_emb_sup=None,
                size_2d=None,
                cross_att_pos=False):
        output = tgt
        output = self.emb_dropout(tgt)
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(output,
                            ref_embs=ref_embs,
                            ref_gt=curr_id_emb,
                            self_pos=self_pos,
                            pos_emb_sup=pos_emb_sup,
                            size_2d=size_2d,
                            cross_att_pos = cross_att_pos)

            if self.return_intermediate:
                intermediate.append(output)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            return intermediate

        return output


class GlobalLocalPATransBlock(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_feedforward=1024,
                 droppath=0.1,
                 global_dropout=0.,
                 local_dropout=0.,
                 patch_num=4,
                 droppath_lst=False,
                 max_dis=7,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, num_head)

        # Global Local-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.global_channelaware_attn = GlobalPatternAware(d_model, use_linear=False, dropout=global_dropout)
        self.local_channelaware_attn = LocalPatternAware(d_model, view_num=patch_num, use_linear=False, dropout=local_dropout)

        self.droppath_lst = droppath_lst

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(
            dim_feedforward, d_model)
    
        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()
        
        self.pos_generator = PositionEmbeddingSine(
            d_model // 2, normalize=True) 

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                ref_embs,
                ref_gt=None,
                self_pos=None,
                pos_emb_sup=None,
                size_2d=(30, 30),
                cross_att_pos=False
                ):
        # Self-attention
        w, bs, d = tgt.shape
        _tgt = self.norm1(tgt) 
        q = k = self.with_pos_embed(_tgt, self_pos) 
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]
        tgt = tgt + self.droppath(tgt2) 
        _tgt = self.norm2(tgt)
        curr_Q = self.linear_Q(_tgt)

        local_Q = curr_Q.permute(1, 2, 0).unsqueeze(1).view(bs,1,d,size_2d[0],size_2d[1])
        curr_Q = curr_Q.permute(1, 0, 2).unsqueeze(2) 

        _ref_embs = self.norm1(ref_embs)
        ref_q = ref_k = self.with_pos_embed(_ref_embs, pos_emb_sup)
        ref_v = _ref_embs
        ref_embs2 = self.self_attn(ref_q, ref_k,ref_v)[0]

        ref_embs = ref_embs + self.droppath(ref_embs2)
        _ref_embs = self.norm2(ref_embs)
        ref_embs = self.linear_Q(_ref_embs)

        support_K = ref_embs.unsqueeze(0)   
        support_V = self.linear_V(ref_embs + ref_gt).unsqueeze(0) 

        if cross_att_pos:
            curr_Q = self.with_pos_embed(curr_Q, self.pos_generator(curr_Q))
            support_K = self.with_pos_embed(support_K, self.pos_generator(support_K))
            support_V = self.with_pos_embed(support_V, self.pos_generator(support_V))
        tgt2 = self.global_channelaware_attn(curr_Q, support_K, support_V)

        _, _, nk, _ = support_K.shape
        support_K = support_K.view(bs, size_2d[0], size_2d[1], nk, d).permute(0,3,4,1,2).contiguous()
        support_V = support_V.view(bs, size_2d[0], size_2d[1], nk, d).permute(0,3,4,1,2).contiguous()

        if cross_att_pos:
            local_Q = self.with_pos_embed(local_Q, self.pos_generator(local_Q))
            support_K = self.with_pos_embed(support_K, self.pos_generator(support_K))
            support_V = self.with_pos_embed(support_V, self.pos_generator(support_V))
        tgt3 = self.local_channelaware_attn(local_Q, support_K, support_V)
        print("tgt", tgt.shape)
        print("tgt2", tgt2.shape)
        print("tgt3", tgt3.shape)
        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3) 
        else:
            tgt = tgt + tgt2 + tgt3

        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt


    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

