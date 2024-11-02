import torch
import torch.nn as nn
import torch.optim as optim
import random
from simple_knn._C import load_param_from_cpu, save_grad_to_cpu

# set random seed
random.seed(0)
torch.manual_seed(0)

# Create synthetic dataset on CPU, pin memory
N = 1000
M = 10000
data_cpu = torch.randn(N, M).pin_memory().requires_grad_(True)

print("data_cpu.sum()", (data_cpu**2).sum().item())

# Initialize the model, loss function, and optimizer; use x as parameters
optimizer = optim.SGD([data_cpu], lr=0.01)

# Training loop with pipelined micro-batches
num_iterations = 10
num_micro_batches = 10  # Number of micro-batches per batch

comm_stream = torch.cuda.Stream(device=0)
default_stream = torch.cuda.current_stream()

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

        data = data_all[i]
        # Forward pass
        load_param_from_cpu(data_cpu, data, batched_sampled_ids[i], N, M)

        # Actual computation
        data2 = data * data
        loss = torch.sum(data2)
        loss.backward()

        save_grad_to_cpu(data_cpu_grad_buffer, data.grad, batched_sampled_ids[i], N, M)

    torch.cuda.synchronize() # this torch cuda synchronize is necessary.
    data_cpu.grad = data_cpu_grad_buffer
    # Update parameters after all micro-batches.
    optimizer.step()

print("data_cpu.sum()", (data_cpu**2).sum().item())

print("Training complete.")
