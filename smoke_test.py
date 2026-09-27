import torch
from models.models_v1 import TeacherStudentNetwork

torch.set_num_threads(2)
torch.manual_seed(0)
for mode in ["parallel", "coupling"]:
    model = TeacherStudentNetwork(dim_feat=32, depth=1, pred_depth=1,
        decoder_depth=1, num_heads=4, num_frames=8, num_joints=25,
        t_patch_size=4, mode=mode)
    output = model(torch.randn(1, 3, 8, 25, 2), mask_ratio=0.5)
    assert len(output) == 5
    assert all(torch.isfinite(v).all() for v in output)
    loss = (output[0]-output[1]).square().mean()
    loss = loss + (output[2]-output[3].detach()).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS:", mode, "synthetic forward and backward")
