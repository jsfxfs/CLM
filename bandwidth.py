import torch
import time

# NOTE:
# To reproduce my bandwidth experiment result on greene, run this:
# > srun --pty --time=4:00:00 -c 1 --mem=40GB --gres=gpu:rtx8000:1 /bin/bash
# > python bandwidth.py

# Define the size of the data
data_size = 1024 * 1024 * 1024  # 4 GB

# Create a tensor on the host (CPU)
h_data = torch.randn(data_size, dtype=torch.float32)

# Allocate memory on the device (GPU)
d_data = torch.empty(data_size, dtype=torch.float32, device='cuda')

bandwidth_h2d = 0
bandwidth_d2h = 0

for i in range(20):
    if i >= 5:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Measure bandwidth from host to device
        torch.cuda.synchronize()
        start_event.record()

        d_data.copy_(h_data.to('cuda'))
        # NOTE: If you call `h_data = h_data.to('cuda')` it takes longer
        # because there's additional memory allocation for a new tensor.
        # So I think using `copy_()`` here is more accurate.

        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time1 = start_event.elapsed_time(end_event)
        bandwidth_h2d = bandwidth_h2d + (data_size * h_data.element_size() * 1000) / (elapsed_time1 * 1024 * 1024 * 1024)  # GB/s

        # Measure bandwidth from device to host
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        h_data.copy_(d_data.to('cpu'))

        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time2 = start_event.elapsed_time(end_event)
        bandwidth_d2h = bandwidth_d2h + (data_size * h_data.element_size() * 1000) / (elapsed_time2 * 1024 * 1024 * 1024)  # GB/s

bandwidth_h2d = bandwidth_h2d / 15
bandwidth_d2h = bandwidth_d2h / 15
        
print(f"Host to Device Bandwidth: {bandwidth_h2d:.2f} GB/s")
print(f"Device to Host Bandwidth: {bandwidth_d2h:.2f} GB/s")

# My experiment result with this script is:
# Host to Device Bandwidth: 4.05 GB/s
# Device to Host Bandwidth: 1.05 GB/s