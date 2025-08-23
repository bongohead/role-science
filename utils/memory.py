# Contains helper functions for CUDA memory management
import torch
import gc
import time 

def check_memory():
    """
    Check memory of all CUDA devices
    """
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found.")
        return

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024/1024/1024:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(i)/1024/1024/1024:.2f} GB")
        print(f"  Total: {torch.cuda.get_device_properties(i).total_memory/1024/1024/1024:.2f} GB")
        print()


def clear_all_cuda_memory(verbose = True):
    """
    Clear all CUDA memory
    """
    # Ensure all CUDA operations are complete
    torch.cuda.synchronize()
    
    # Empty the cache on all devices
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Clear references to any tensors and force garbage collection
    gc.collect()
    
    # Optionally, reset the CUDA context (commented out as it's more drastic and may not always be necessary)
    # for device_id in range(torch.cuda.device_count()):
    #     torch.cuda.reset()
    if verbose:
        print("All CUDA memory cleared on all devices.")


def profile_memory(func, warmup = 3, runs = 10, *args, **kwargs):
    """
    Profile peak CUDA memory usage of a torch function. Uses warmup/multiple passes for more accuracy.

    Params
        @func: The function to test.
        @warmup: Number of warmup runs before timing.
        @runs: Number of timed runs.
        @args, kwarsg: Arguments to pass to the function
    
    Examples:
        profile_memory(np.diff, a = [0, 2, 5, 10, 12], n = 1)
    """
    for _ in range(warmup):
        func(*args, **kwargs)
        torch.cuda.synchronize()  # Make sure each warmup run finishes

    times = []
    peak_mems = []
    incd_mens = []

    for _ in range(runs):
        # Clear caches & reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure allocated memory before
        start_mem_bytes = torch.cuda.memory_allocated()
        
        # Start timing
        start_time = time.time()
        
        # Run the function (forward + backward)
        result = func(*args, **kwargs)
        
        # Synchronize to ensure all GPU work completes
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Measure memory usage after
        end_mem_bytes = torch.cuda.memory_allocated()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        
        times.append(end_time - start_time)
        
        peak_mems.append(peak_mem_bytes)
        incd_mens.append(end_mem_bytes - start_mem_bytes)
        
    avg_time = sum(times)/len(times)
    avg_peak_mem = sum(peak_mems)/len(peak_mems)
    avg_incd_mem = sum(incd_mens)/len(incd_mens)

    return {
        "runs": runs,
        "average_time":  f"{avg_time:.8f}s",
        "average_peak_mem": f"{(avg_peak_mem/1e6):.4f}MB",
        "average_increase_mem_MB": f"{(avg_incd_mem/1e6):.4f}MB",
    }