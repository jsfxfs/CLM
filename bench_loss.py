# torchrun --standalone --nnodes=1 --nproc-per-node=4 test_compiled_loss.py
# nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop -o bench_loss python /global/homes/j/jy-nyu/refactor/Grendel-GS-internal/bench_loss.py

import torch
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import init_distributed
import time
from fused_ssim import fused_ssim

init_distributed(None)

class FinalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_dssim = 0.2 # TODO: allow this to be set by the user
    
    def forward(self, image, image_gt_original):
        image_gt = torch.clamp(image_gt_original / 255.0, 0.0, 1.0)
        Ll1 = l1_loss(image, image_gt)
        ssim_loss = ssim(image, image_gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (
                1.0 - ssim_loss
            )
        return loss

class FusedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_dssim = 0.2 # TODO: allow this to be set by the user
    
    def forward(self, image, image_gt_original):
        image_gt = torch.clamp(image_gt_original / 255.0, 0.0, 1.0)
        Ll1 = l1_loss(image, image_gt)
        image = image.unsqueeze(0)
        image_gt = image_gt.unsqueeze(0)
        ssim_loss = fused_ssim(image, image_gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (
                1.0 - ssim_loss
            )
        return loss

@torch.compile
def loss_combined(image, image_gt, ssim_loss):
    lambda_dssim = 0.2 # TODO: allow this to be set by the user
    Ll1 = l1_loss(image, image_gt)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (
                1.0 - ssim_loss
            )
    return loss

class FusedCompiledLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, image_gt_original):
        image_gt = torch.clamp(image_gt_original / 255.0, 0.0, 1.0)
        ssim_loss = fused_ssim(image.unsqueeze(0), image_gt.unsqueeze(0))
        return loss_combined(image, image_gt, ssim_loss)

UNCOMPILED_LOSS_MODULE = FinalLoss()
COMPILED_LOSS_MODULE = torch.compile(FinalLoss())
FUSED_LOSS_MODULE = FusedLoss()
FUSED_COMPILED_LOSS_MODULE = FusedCompiledLoss()

device = "cuda"

# cudaProfilerApi
torch.cuda.cudart().cudaProfilerStart()

image = torch.randn(3, 1080, 1920, device=device)
# image = torch.randn(3, 256, 256, device=device)
image_gt_original = torch.randn(*image.shape, device=device)

warmup_iterations = 5
iterations = 10

torch.cuda.synchronize()
torch.cuda.nvtx.range_push("Compiled loss")
for i in range(warmup_iterations+iterations):
    if i == warmup_iterations:
        start_time = time.time()
    loss = COMPILED_LOSS_MODULE(image, image_gt_original)
print("Compiled loss: ", loss.item())
torch.cuda.synchronize()
end_time = time.time()
print("Compiled loss time: ", end_time - start_time)
torch.cuda.nvtx.range_pop()


torch.cuda.synchronize()
torch.cuda.nvtx.range_push("Uncompiled loss")
for i in range(warmup_iterations+iterations):
    if i == warmup_iterations:
        start_time = time.time()
    loss = UNCOMPILED_LOSS_MODULE(image, image_gt_original)
print("Uncompiled loss: ", loss.item())
torch.cuda.synchronize()
end_time = time.time()
print("Uncompiled loss time: ", end_time - start_time)
torch.cuda.nvtx.range_pop()

torch.cuda.synchronize()
torch.cuda.nvtx.range_push("Fused loss")
for i in range(warmup_iterations+iterations):
    if i == warmup_iterations:
        start_time = time.time()
    loss = FUSED_LOSS_MODULE(image, image_gt_original)
print("Fused loss: ", loss.item())
torch.cuda.synchronize()
end_time = time.time()
print("Fused loss time: ", end_time - start_time)
torch.cuda.nvtx.range_pop()


torch.cuda.synchronize()
torch.cuda.nvtx.range_push("Fused Compiled loss")
for i in range(warmup_iterations+iterations):
    if i == warmup_iterations:
        start_time = time.time()
    loss = FUSED_COMPILED_LOSS_MODULE(image, image_gt_original)
print("Fused Compiled loss: ", loss.item())
torch.cuda.synchronize()
end_time = time.time()
print("Fused Compiled loss time: ", end_time - start_time)
torch.cuda.nvtx.range_pop()

print("Image Shape in these losses: ", image.shape)

torch.cuda.cudart().cudaProfilerStop()