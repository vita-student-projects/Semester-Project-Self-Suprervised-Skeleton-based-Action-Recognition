import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from model.blocks import SpatialFormer, TemporalFormer, Predictor, ActionHeadLinprobe, ActionHeadFinetune
from model.utils import to_device, to_var


class StudentMLPNetwork(nn.Module):

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
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.num_classes = num_classes

        self.studentspatial = TemporalFormer(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.depth, 
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
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def patchify(self, imgs):
        """
        imgs: (N, T, V, 3)
        x: (N, L, t_patch_size * patch_size * 3)
        """
        NM, T, V, C = imgs.shape
        p = self.patch_size
        u = self.t_patch_size
        assert V % p == 0 and T % u == 0
        VP = V // p
        TP = T // u

        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))
        x = torch.einsum("ntuvpc->ntvupc", x)
        x = x.reshape(shape=(NM, TP, VP, u * p * C))
        return x

    def forward(self, x):
        B, M, T, J, dim_in = x.shape
        x = x.contiguous().view(B * M, T, J, dim_in)
        x = self.patchify(x)
        B_, T_, J_, dim_in = x.shape
        studentencoded = self.studentspatial(x)

        studentencoded = studentencoded.reshape(B, M, T_, J_, -1)
        # print(studentencoded.shape)
        pred = self.head(studentencoded)

        return pred

    