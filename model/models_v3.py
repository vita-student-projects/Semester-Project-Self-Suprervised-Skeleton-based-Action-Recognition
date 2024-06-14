import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from model.blocks import SpatialFormer, TemporalFormer, Predictor, Decoder
from model.utils import to_device, to_var


class TeacherStudentNetwork(nn.Module):

    def __init__(self, dim_in=3, dim_feat=256, depth=5, pred_depth=3, num_heads=8, mlp_ratio=4, 
                    num_frames=100, num_joints=25, patch_size=1, t_patch_size=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                    norm_layer=nn.LayerNorm,
                    alpha=0.9, mask_ratio=0.):
        super().__init__()

        self.dim_in = dim_in
        self.dim_feat = dim_feat
        self.depth = depth
        self.pred_depth = pred_depth
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
        self.mask_ratio = mask_ratio

        self.studentspatial = SpatialFormer(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints, 
                                        patch_size=self.patch_size, t_patch_size=self.t_patch_size,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)
        self.teacherspatial = SpatialFormer(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints, 
                                        patch_size=self.patch_size, t_patch_size=self.t_patch_size,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)

        self.alpha = alpha
        self.predictor = Predictor(dim_in=self.dim_feat, dim_feat=self.dim_feat, depth=self.pred_depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)
        self.decoder = Decoder(dim_in=self.dim_feat, dim_feat=self.dim_feat, depth=self.pred_depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints,
                                        patch_size=self.patch_size, t_patch_size=self.t_patch_size,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)
        self.decoder_pred = nn.Linear(
            dim_feat,
            t_patch_size * patch_size * dim_in,
            bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_feat))
        def _no_grad_trunc_normal_(tensor, mean, std, a, b):
            def norm_cdf(x):
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                            "The distribution of values may be incorrect.",
                            stacklevel=2)
            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor

        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            # type: (Tensor, float, float, float, float) -> Tensor
            return _no_grad_trunc_normal_(tensor, mean, std, a, b)
        trunc_normal_(self.mask_token, std=.02)

        for param_stu, param_tea in zip(
            self.studentspatial.parameters(), self.teacherspatial.parameters()
        ):
            param_tea.data.copy_(param_stu.data)  # initialize
            param_tea.requires_grad = False  # not update by gradient
            
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
    
    @torch.no_grad()
    def _ema_update_teacher_encoder(self):
        teacher_params = self.teacherspatial.parameters()
        student_params = self.studentspatial.parameters()
        for teacher_param, student_param in zip(teacher_params, student_params):
            teacher_param.data = self.alpha * teacher_param.data + (1-self.alpha) * student_param.data
    


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, T, J, dim_in = x.shape
        x = x.reshape(B, T*J, dim_in)
        N, L, D = x.shape  # batch, length, dim
        len_keep = math.ceil(L * (1 - mask_ratio) / T) * T

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward(self, x):
        
        B, M, T, J, dim_in = x.shape
        x = x.contiguous().view(B * M, T, J, dim_in)
        x = self.patchify(x)
        x_patch = x
        
        B, T, J, dim_in = x.shape

        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        _, masked_dim, _ = x_masked.shape
        x_masked = x_masked.reshape(B, T, -1, dim_in)
        studentencoded = self.studentspatial(x_masked)
        _, Tp, Jp, _ = studentencoded.shape
        teacherencoded = self.teacherspatial(x)

        # student -> Predictor
        # teacher -> Masked

        mask_tokens = self.mask_token.repeat(B, T * J - masked_dim, 1)
        studentencoded_ = studentencoded.reshape(B, -1, self.dim_feat)
        studentencoded_ = torch.cat([studentencoded_[:, :, :], mask_tokens], dim=1)
        studentencoded_ = torch.gather(
            studentencoded_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, studentencoded_.shape[2])
        )  # unshuffle
        studentencoded = studentencoded_.view([B, T, J, self.dim_feat])

        studentpredict = self.predictor(studentencoded)
        teacherdecode = self.decoder(teacherencoded)
        teacherproject = self.decoder_pred(teacherdecode)
        
        studentpredict_ = studentpredict.reshape(B, -1, self.dim_feat)
        teacherdecoded_ = teacherdecode.reshape(B, -1, self.dim_feat)

        studentpredict = studentpredict_.masked_fill((1-mask).unsqueeze(-1).bool(), 0)
        teachermasked = teacherdecoded_.masked_fill((1-mask).unsqueeze(-1).bool(), 0)

        return studentencoded, teacherencoded, teacherproject, x_patch, studentpredict, teachermasked, mask

    