import torch
import torch.nn as nn
import torch.nn.functional as F
from patrans_core.PATrans_module.layers.basic import DropOutLogit
import time
import os

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head=8, dropout=0., use_linear=True):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.hidden_dim = d_model // num_head
        self.T = (d_model / num_head)**0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V):
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        # Scale
        Q = Q / self.T

        # Multi-head
        Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3) 
        K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 3, 0) 
        V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3) 

        # Multiplication
        QK = Q @ K 

        # Activation
        attn = torch.softmax(QK, dim=-1)

        # Dropouts
        attn = self.dropout(attn) 
        
        # Weighted sum
        outputs = (attn @ V).permute(2, 0, 1, 3) 

        # Restore shape
        outputs = outputs.reshape(-1, bs, self.d_model) 
        outputs = self.projection(outputs)
  
        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class GlobalPatternAware(nn.Module):
    def __init__(self, d_model, dropout=0., use_linear=True):
        super().__init__()
        self.d_model = d_model
        self.T = d_model ** 0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V):
        # q: target_img
        # k: support_img
        # v: support_img + support_gt
        
        bs, w, n, d = Q.size()
        _, _, nk, _ = K.size()
        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        # Scale
        Q = Q / self.T
        
        Q = Q.permute(0, 3, 2, 1).view(bs*d, n, w) 
        K = K.permute(0, 3, 2, 1).view(bs*d, nk, w) 
        V = V.permute(0, 3, 2, 1).view(bs*d, nk, w) 

        # pattern-aware
        QK = torch.bmm(Q, K.transpose(1,2)) 

        # Activation
        attn = torch.softmax(QK, dim=2) 

        # Dropouts
        attn = self.dropout(attn)

        # Weighted sum
        outputs = torch.bmm(attn, V) 

        # Restore shape
        outputs = outputs.reshape(bs, d, w).permute(2, 0, 1).contiguous()
        outputs = self.projection(outputs) 

        return outputs

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LocalPatternAware(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.,
                 view_num=4,
                 dilation=1,
                 use_linear=True):
        super().__init__()
        self.dilation = dilation
        self.view_num = view_num
        self.view_num_sqrt = int(view_num**0.5)

        self.T = d_model**0.5

        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_K = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_V = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.projection = nn.Linear(d_model, d_model)
        self.dropout = DropOutLogit(dropout)

        self.padded_local_mask = None
        self.local_mask = None

    def forward(self, q, k, v):
        # q: target_img
        # k: support_img
        # v: support_img + support_gt
        q_shape3_padding = False
        q_shape4_padding = False
        if q.shape[3] % 2 == 1:
            q = F.pad(q, (0, 0, 1, 0, 0, 0), mode='replicate')
            k = F.pad(k, (0, 0, 1, 0, 0, 0), mode='replicate')
            v = F.pad(v, (0, 0, 1, 0, 0, 0), mode='replicate')
            q_shape3_padding = True

        if q.shape[4] % 2 == 1:
            q = F.pad(q, (1, 0, 0, 0, 0, 0), mode='replicate')
            k = F.pad(k, (1, 0, 0, 0, 0, 0), mode='replicate')
            v = F.pad(v, (1, 0, 0, 0, 0, 0), mode='replicate')
            q_shape4_padding = True
        bs, n, c, h, w = q.size() 
        bs, nk, _, _, _ = k.size()

        if self.use_linear:
            q = q.view(bs*n, c, h, w)
            k = k.view(bs*nk, c, h, w)
            v = v.view(bs*nk, c, h, w)
            q = self.linear_Q(q).view(bs, n, c, h, w).contiguous()
            k = self.linear_K(k).view(bs, n, c, h, w).contiguous()
            v = self.linear_V(v).view(bs, n, c, h, w).contiguous()

        # Scale
        q = q / self.T
        
        local_out = torch.ones(bs, c, h, w).to(q.device)     

        self.window_h = int(h / self.view_num_sqrt) 
        self.window_w =int(w / self.view_num_sqrt)    
        for i in range(0, self.view_num_sqrt):
            for j in range(0, self.view_num_sqrt):
                patch_q = q[:,:,:,i*self.window_h:(i+1)*self.window_h,j*self.window_w:(j+1)*self.window_w].permute(0,2,1,3,4).contiguous().view(bs*c,n,self.window_h*self.window_w) #[bs*96, 1, window_h*window_w]
                patch_k = k[:,:,:,i*self.window_h:(i+1)*self.window_h,j*self.window_w:(j+1)*self.window_w].permute(0,2,1,3,4).contiguous().view(bs*c,nk,self.window_h*self.window_w) #[bs*96, 2, window_h*window_w]
                patch_v = v[:,:,:,i*self.window_h:(i+1)*self.window_h,j*self.window_w:(j+1)*self.window_w].permute(0,2,1,3,4).contiguous().view(bs*c,nk,self.window_h*self.window_w) #[bs*96, 2, window_h*window_w]

                patch_qk = torch.bmm(patch_q, patch_k.transpose(1,2)) 
                patch_attn = torch.softmax(patch_qk, dim=2)     
                patch_attn = self.dropout(patch_attn)    
                patch_out = torch.bmm(patch_attn, patch_v) 
                local_out[:,:,i*self.window_h:(i+1)*self.window_h, j*self.window_w:(j+1)*self.window_w]=patch_out.contiguous().view(bs,c,self.window_h, self.window_w)
        if q_shape3_padding:
            local_out = local_out[:,:,1:,:]
            h = h-1
        if q_shape4_padding:
            local_out = local_out[:,:,:,1:]
            w = w-1
            
        multiview_out = local_out.contiguous().view(bs, c, h*w).permute(2,0,1) 
        output = self.projection(multiview_out)

        return output
