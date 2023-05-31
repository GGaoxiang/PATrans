import torch.nn as nn
from PATrans_module.layers.transformer import GlobalLocalPATrans
from PATrans_module.layers.position import PositionEmbeddingSine


class PATrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128

        self.globallocalpatrans = GlobalLocalPATrans(
            num_layers=1, 
            d_model=self.dim, 
            num_head=8,
            emb_dropout=0., 
            droppath=0.1, 
            global_dropout=0., 
            local_dropout=0., 
            patch_num = 4, 
            max_dis = 7,
            droppath_lst=False, 
            droppath_scaling=False, 
            intermediate_norm=True, 
            cross_att_pos = False,
            return_intermediate=True
            )

        self.pos_generator = PositionEmbeddingSine(
            self.dim // 2, normalize=True) 

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def PATrans_forward(self,
                     curr_embs,
                     ref_embs,
                     ref_gt,
                     pos_emb=None,
                     pos_emb_sup=None,
                     size_2d=(30, 30)):
        n_ref, c, h, w = ref_embs.size()
        curr_emb = curr_embs.view(1, c, h * w).permute(2, 0, 1) 
        ref_embs = ref_embs.view(n_ref, c, h * w).permute(2, 0, 1)  
        
        patrans_embs = self.globallocalpatrans(curr_emb, ref_embs, ref_gt, pos_emb, pos_emb_sup, size_2d)
     
        return patrans_embs[0]

