import torch.nn as nn
from patrans_core.PATrans_module.layers.transformer import GlobalLocalPATrans
from patrans_core.PATrans_module.layers.position import PositionEmbeddingSine


class PATrans(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL.TRANS.MODEL_MAX_OBJ_NUM 
        self.epsilon = cfg.MODEL.TRANS.MODEL_EPSILON 

        self.globallocalpatrans = GlobalLocalPATrans(
            cfg.MODEL.TRANS.MODEL_PATRANS_NUM, 
            dim, 
            num_head=cfg.MODEL.TRANS.MODEL_SELF_HEADS,
            emb_dropout=cfg.MODEL.TRANS.TRAIN_PATRANS_EMB_DROPOUT, 
            droppath=cfg.MODEL.TRANS.TRAIN_PATRANS_DROPPATH, 
            global_dropout=cfg.MODEL.TRANS.TRAIN_PATRANS_GLOBAL_DROPOUT, 
            local_dropout=cfg.MODEL.TRANS.TRAIN_PATRANS_LOCAL_DROPOUT, 
            patch_num = cfg.MODEL.TRANS.PATCH_NUM, 
            max_dis = cfg.MODEL.TRANS.MAX_DIS,
            droppath_lst=cfg.MODEL.TRANS.TRAIN_PATRANS_DROPPATH_LST, 
            droppath_scaling=cfg.MODEL.TRANS.TRAIN_PATRANS_DROPPATH_SCALING, 
            intermediate_norm=cfg.MODEL.TRANS.MODEL_DECODER_INTERMEDIATE_PATRANS, 
            cross_att_pos = cfg.MODEL.TRANS.MODEL_CROSS_ATT_COSPOSITION,
            return_intermediate=True
            )

        self.pos_generator = PositionEmbeddingSine(
            dim // 2, normalize=True) 

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

