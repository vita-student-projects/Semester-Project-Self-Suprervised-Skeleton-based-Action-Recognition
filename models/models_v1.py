import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from models.blocks import Former, Predictor, Decoder
import torch.distributed as dist

class TeacherStudentNetwork(nn.Module):

    def __init__(self, dim_in=3, dim_feat=256, depth=5, pred_depth=3, decoder_depth=3, num_heads=8, mlp_ratio=4, 
                    num_frames=100, num_joints=25, patch_size=1, t_patch_size=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                    norm_layer=nn.LayerNorm, mode='parallel',
                    alpha=0.9999):
        super().__init__()

        self.dim_in = dim_in
        self.dim_feat = dim_feat
        self.depth = depth
        self.pred_depth = pred_depth
        self.decoder_depth = decoder_depth
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
        self.mode = mode

        self.studentParallel = Former(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints, 
                                        patch_size=self.patch_size, t_patch_size=self.t_patch_size,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer,
                                        mode=self.mode)
        self.teacherParallel = Former(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints, 
                                        patch_size=self.patch_size, t_patch_size=self.t_patch_size,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer,
                                        mode=self.mode)
        self.alpha = alpha
        self.predictor = Predictor(dim_in=self.dim_feat, dim_feat=self.dim_feat, depth=self.pred_depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)
        
        self.decoder = Decoder(dim_in=self.dim_in, dim_feat=self.dim_feat, depth=self.decoder_depth, 
                                        num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, num_frames=self.num_frames, num_joints=self.num_joints,
                                        patch_size=self.patch_size, t_patch_size=self.t_patch_size,
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop_rate=self.drop_rate, 
                                        attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate, norm_layer=self.norm_layer)

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
            self.studentParallel.parameters(), self.teacherParallel.parameters()
        ):
            param_tea.data.copy_(param_stu.data)  # initialize
            param_tea.requires_grad = False  # not update by gradient

        self.adjust_para = nn.Parameter(torch.tensor(1.0))
    

    @torch.no_grad()
    def _ema_update_teacher_encoder(self):
        teacher_named_params = self.teacherParallel.named_parameters()
        student_named_params = self.studentParallel.named_parameters()
        # result = 0
        for (teacher_name, teacher_param), (student_name, student_param) in zip(teacher_named_params, student_named_params):
            teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)
            

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
        x = x.reshape(shape=(NM, TP * VP, u * p * C))
        return x


    def forward(self, x, mask_ratio=0.9, motion_stride=1, motion_aware_tau=0.8):
        N, C, T, J, M = x.shape

        studentEncoded, mask, ids_restore, x_motion = self.studentParallel(
                                x, mask_ratio=mask_ratio, motion_stride=motion_stride, 
                                motion_aware_tau=motion_aware_tau, setting='student')
        _, T_s, J_s, _ = studentEncoded.shape

        teacherEncoded, _, _, x_motion = self.teacherParallel(
                                x, mask_ratio=0, motion_stride=motion_stride, 
                                motion_aware_tau=motion_aware_tau, setting='teacher')

        mask_tokens = self.mask_token.repeat(M*N, T//self.t_patch_size-T_s, J//self.patch_size, 1)
        studentEncoded_ = torch.cat([studentEncoded[:, :, :, :], mask_tokens], dim=1)
        studentEncoded_ = torch.gather(
            studentEncoded_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, studentEncoded_.shape[3])
        )
        studentEncodedMaskToken = studentEncoded_.view([M*N, T//self.t_patch_size, J//self.patch_size, self.dim_feat])

        # student -> Predictor, Masked, Decoder
        # teacher -> Masked

        studentPredict = self.predictor(studentEncodedMaskToken)
        studentProject = self.decoder(studentEncodedMaskToken)
        
        studentMotionPredict = studentPredict.masked_fill((1-mask).unsqueeze(-1).bool(), 0)
        teacherMotionGT = teacherEncoded.masked_fill((1-mask).unsqueeze(-1).bool(), 0)

        target = self.patchify(x_motion)
        return studentProject, target, studentMotionPredict, teacherMotionGT, mask
        # return studentEncoded, teacherEncoded, studentEncodedMaskToken, studentProject, target, studentMotionPredict, teacherMotionGT, mask


if __name__ == "__main__":
    B = 4
    T = 100
    J = 25
    C = 3
    M = 2
    x = torch.rand(B, C, T, J, M)
    print(x.shape)
    model = TeacherStudentNetwork()
    studentEncoded, teacherEncoded, studentEncodedMaskToken, studentProject, target, studentMotionPredict, teacherMotionGT, mask = model(x)
    print(studentEncoded.shape)
    print(teacherEncoded.shape)
    print(studentEncodedMaskToken.shape)
    print(studentProject.shape)
    print(target.shape)
    print(studentMotionPredict.shape)
    print(teacherMotionGT.shape)
    print(mask.shape)
    print(torch.sum(mask, dim=1))