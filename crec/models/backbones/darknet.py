# coding=utf-8

import torch.nn as nn

from crec.layers.blocks import darknet_conv, darknetblock
from crec.utils.parse_darknet_weights import parse_yolo_weights

class DarkNet53(nn.Module):
    def __init__(
        self, 
        pretrained_weight_path, 
        pretrained=False, 
        multi_scale_outputs=False,
        freeze_backbone=False,
    ):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(darknet_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
        self.module_list.append(darknet_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=64))
        self.module_list.append(darknet_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=128, nblocks=2))
        self.module_list.append(darknet_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=256, nblocks=8))  # shortcut 1 from here
        self.module_list.append(darknet_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=512, nblocks=8))  # shortcut 2 from here
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=1024, nblocks=4))

        # YOLOv3
        self.module_list.append(darknetblock(ch=1024, nblocks=2, shortcut=False))
        self.module_list.append(darknet_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
        # 1st yolo branch
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        self.multi_scale_outputs=multi_scale_outputs
        
        if pretrained:
            parse_yolo_weights(self, pretrained_weight_path)

        if freeze_backbone:
            self.frozen(self.module_list[:-2])

    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        outputs=[]
        for i,module in enumerate(self.module_list):
            x=module(x)
            if i in [6,8]:
                outputs.append(x)
        outputs.append(x)
        if self.multi_scale_outputs:
            return outputs
        else:
            return x