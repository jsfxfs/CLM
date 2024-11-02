import torch
import torch.nn as nn
import torch.optim as optim
import random
from simple_knn._C import load_param_from_cpu, save_grad_to_cpu, load_param_from_cpu_stream, save_grad_to_cpu_stream

# set random seed
random.seed(0)
torch.manual_seed(0)

# Create synthetic dataset on CPU, pin memory
N = 1000
M = 10000
data_cpu = (torch.randn(N, M)**2).pin_memory().requires_grad_(True)

print("data_cpu.sum()", (data_cpu**2).sum().item())

# Initialize the model, loss function, and optimizer; use x as parameters
optimizer = optim.SGD([data_cpu], lr=0.01)

# Training loop with pipelined micro-batches
num_iterations = 10
num_micro_batches = 10  # Number of micro-batches per batch

comm_stream = torch.cuda.Stream(device=0)
default_stream = torch.cuda.current_stream()

# print stream information
print("comm_stream: ", comm_stream)
print("default_stream: ", default_stream)

for iteration_idx in range(num_iterations):

    # sample num_micro_batches
    sampled_ids = torch.randperm(N, dtype=torch.int, device='cuda')
    batched_sampled_ids = sampled_ids.split(N//num_micro_batches)

    data_cpu_grad_buffer = torch.zeros(N, M).pin_memory()

    # Declare torch.tensor for data on GPU
    data_all = []
    for i in range(num_micro_batches):
        data_all.append(torch.empty((batched_sampled_ids[i].shape[0], M),
                                    dtype=torch.float32,
                                    device='cuda',
                                    requires_grad=True))

    # Accumulate gradients over micro-batches
    optimizer.zero_grad()

    for i in range(num_micro_batches):

        # with torch.cuda.stream(comm_stream):

        # make data to be a leaf variable which can receive gradients
        data = data_all[i]

        with torch.cuda.stream(comm_stream):
            # Forward pass
            load_param_from_cpu_stream(data_cpu, data, batched_sampled_ids[i], N, M)
            # create an event
            cpu2gpu_event = torch.cuda.Event(enable_timing=True)
            cpu2gpu_event.record(comm_stream)

        # sync event of comm_stream with default_stream
        cpu2gpu_event.wait(default_stream)
        
        # Actual computation, matrix multiplication data*data^T
        data_sq2 = torch.mm(data, data.t())
        loss = torch.sum(data_sq2) / batched_sampled_ids[i].shape[0] / batched_sampled_ids[i].shape[0]
        loss.backward()

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        with torch.cuda.stream(comm_stream):
            gpu2cpu_event.wait(comm_stream)
            # sync event of default_stream with comm_stream
            save_grad_to_cpu_stream(data_cpu_grad_buffer, data.grad, batched_sampled_ids[i], N, M)

    torch.cuda.synchronize() # this torch cuda synchronize is necessary.
    data_cpu.grad = data_cpu_grad_buffer
    # Update parameters after all micro-batches.
    optimizer.step()

print("data_cpu.sum()", (data_cpu**2).sum().item())

print("Training complete.")

# module load pytorch/2.1.0-cu12
# cd /pscratch/sd/j/jy-nyu/ && # now we do not need this

# nsys profile --force-overwrite true -o test_pipeliningv2 python test_pipeliningv2.py
## nsys profile -t cuda,nvtx python test_pipeliningv2.py
# nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o test_pipeliningv2 python test_pipeliningv2.py