import torch
from torch.profiler import profile, ProfilerActivity, record_function

assert torch.cuda.is_available()

# Allocate x before profiling so its creation doesn't confuse the trace.
x = torch.ones(3, device="cuda")
torch.cuda.synchronize()

# Ask the profiler to record CUDA synchronization calls.
torch._C._profiler._set_cuda_sync_enabled_val(True)

results = []

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:

    with record_function("01_full_cuda"):
        results.append(torch.full((3,), 1.0, device="cuda"))

    with record_function("02_arange_cuda"):
        results.append(torch.arange(3, device="cuda"))

    with record_function("03_add_python_scalar"):
        results.append(x + 1.0)

    with record_function("04_add_cpu_0dim"):
        results.append(x + torch.tensor(1.0))

    with record_function("05_tensor_from_python_list"):
        results.append(git 
            torch.tensor([0.0, 1.0, 2.0], device="cuda")
        )

    # Clearly label the synchronization used to finish profiling.
    with record_function("99_explicit_final_sync"):
        torch.cuda.synchronize()

prof.export_chrome_trace("/content/hidden_h2d_trace.json")

print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=100,
    )
)