import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from model.blocks import SpatialFormer, TemporalFormer, Predictor, ActionHeadLinprobe, ActionHeadFinetune, VanillaFormer
from model.utils import to_device, to_var


class SVMLPNetwork(nn.Module):

    def __init__(self, dim_in=3, dim_feat=256, depth=5, num_heads=8, mlp_ratio=4, 
                    num_frames=100, num_joints=25, patch_size=1, t_patch_size=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                    norm_layer=nn.LayerNorm, protocol='finetune', num_classes=60):
        super().__init__()

        self.dim_in = dim_in
        self.dim_feat = dim_feat
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.num_classes = num_classes

        self.studentspatial = VanillaFormer(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints, 
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)
        if protocol == 'linprobe':
            self.head = ActionHeadLinprobe(dim_feat=dim_feat, num_classes=num_classes)
        elif protocol == 'finetune':
            self.head = ActionHeadFinetune(dropout_ratio=0.3, dim_feat=dim_feat, num_classes=num_classes)
        else:
            raise TypeError('Unrecognized evaluation protocol!')

        for param in self.studentspatial.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True


    def forward(self, x):
        B, M, T, J, dim_in = x.shape
        x = x.contiguous().view(B * M, T, J, dim_in)
        B_, T_, J_, dim_in = x.shape
        vanillaformer = self.studentspatial(x)

        vanillaencoded = vanillaformer.reshape(B, M, T, J, -1)
        pred = self.head(vanillaencoded)

        return pred

    