# coding=utf-8

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled=False

class CREC(nn.Module):
    def __init__(
        self, 
        visual_backbone: nn.Module, 
        language_encoder: nn.Module,
        multi_scale_manner: nn.Module,
        fusion_manner: nn.Module,
        attention_manner: nn.Module,
        head: nn.Module,
    ):
        super(CREC, self).__init__()
        self.visual_encoder=visual_backbone
        self.lang_encoder=language_encoder
        self.multi_scale_manner = multi_scale_manner
        self.fusion_manner = fusion_manner
        self.attention_manner = attention_manner
        self.head=head
        self.ori_dataset=False
        self.visual_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512))
        self.cl_loss=nn.CrossEntropyLoss()
        
    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, x, y, aw=None, neg=None, det_label=None, alpha=0.5, temp=0.2):        
        # vision and language encoding
        bs = y.shape[0]
        x = self.visual_encoder(x)                        # [[bs, 256, x, x], [bs, 512, x/2, x/2], [bs, 1024, x/4, x/4]
        y = self.lang_encoder(y)['flat_lang_feat']

        if aw is not None:
            aw = self.lang_encoder(aw)['flat_lang_feat']
        
        if neg is not None:
            n_y = self.lang_encoder(neg)['flat_lang_feat']
    
        # vision and language fusion
        fused_x = []
        for i in range(len(self.fusion_manner)):          # [[bs,512,x,x], [bs,512,x/2,x/2], [bs,512,x/4,x/4]]
            fused_x.append(x[i].clone())
            fused_x[i] = self.fusion_manner[i](fused_x[i], y)
        fused_x = self.multi_scale_manner(fused_x)

        # multi-scale fusion
        vt_feats, _, _ =self.attention_manner(y, fused_x[-1])
        if aw != None:
            va_feats, _, _ =self.attention_manner(aw, fused_x[-1])
            cf_feats = alpha * vt_feats + (1-alpha) * va_feats
        else:
            cf_feats = vt_feats
        
        # contrastive learning   
        # calculate the distance between fusion feature and textual features (y: f_T+, n_y: f_T-)
        if self.training and neg is not None:
            ks = math.floor(vt_feats.shape[2]/2.0)
            pooled = F.avg_pool2d(vt_feats, ks, ks)

            f_v = self.visual_proj(pooled).view(bs, 1, -1)
            f_t = torch.hstack((y.view(bs, 1, -1), n_y.view(bs, 1, -1)))
            f_t = f_t.transpose(1,2)
        
            logits = torch.bmm(f_v, f_t)
            logits = F.softmax(logits.view(bs, -1), dim=1)
            cl_label = det_label[:, :, 4].view(-1).to(torch.int64)      # cf_id=0 -> 0th is postive

        # return
        if self.training:
            gamma_det = 1
            gamma_cls = 2
            gamma_cl = 2

            loss_det, loss_cls = self.head(vt_feats, cf_feats, labels=det_label)
            if neg is not None:
                loss_cl = self.cl_loss(logits/temp, cl_label)
            else:
                loss_cl = 0
            
            return gamma_det*loss_det + gamma_cls*loss_cls + gamma_cl*loss_cl, loss_det, loss_cls, loss_cl
        else:
            box, cf_preds = self.head(vt_feats, cf_feats)
            return box, cf_preds
