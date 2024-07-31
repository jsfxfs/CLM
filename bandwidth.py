import torch
import time

# Define the size of the data
data_size = 1024 * 1024 * 1024  # 4 GB

### Measure bandwidth from host to device

# Allocate memory on the device (GPU)
d_data = torch.empty(data_size, dtype=torch.float32, device='cuda')
# Create a tensor on the host (CPU)
h_data = torch.randn(data_size, dtype=torch.float32).pin_memory()

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
bandwidth_h2d = (data_size * h_data.element_size() * 1000) / (elapsed_time1 * 1024 * 1024 * 1024)  # GB/s






### Measure bandwidth from device to host
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

d_data = torch.randn(data_size, dtype=torch.float32).pin_memory() # Save it in pinned memory
d_data = d_data.to('cuda') # We have to move the pinned memory to GPU

torch.cuda.synchronize()
start_event.record()

d_data = d_data.to('cpu', non_blocking=True)
torch.cuda.current_stream().synchronize()
# reference: https://discuss.pytorch.org/t/how-to-maximize-cpu-gpu-memory-transfer-speeds/173855/4
# Please read this for more information

torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
elapsed_time2 = start_event.elapsed_time(end_event)


bandwidth_d2h = (data_size * h_data.element_size() * 1000) / (elapsed_time2 * 1024 * 1024 * 1024)  # GB/s
        
print(f"Host to Device Bandwidth: {bandwidth_h2d:.2f} GB/s")
print(f"Device to Host Bandwidth: {bandwidth_d2h:.2f} GB/s")


# On Perlmutter: PCIe 4.0 * 16 ----> theoretically 31.5 GB/s 
# (cpu_adam) jy-nyu@nid200257:~/cpu_adam/benchmark_cpu2gpu> python benchmark_cpu2gpu.py 
# h_data.element_size(): 4
# Host to Device Bandwidth: 23.47 GB/s
# Device to Host Bandwidth: 24.53 GB/s

