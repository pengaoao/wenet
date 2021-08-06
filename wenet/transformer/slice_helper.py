import torch

@torch.jit.script
def slice_helper(x, offset):
  return x[:, -offset: , : ]

@torch.jit.script
def slice_helper2(x, start, end):
    return x[:, start:end]

@torch.jit.script
def slice_helper3(x, start):
  return x[:, start:]

@torch.jit.script
def get_item(x):
  item = x.detach().item()
  output = torch.tensor(item)
  return output

@torch.jit.script
def get_next_cache_start(required_cache_size: torch.Tensor, xs: torch.Tensor):
  # required_cache_size = required_cache_size_tensor.detach().item()
  next_cache_start = 0
  if required_cache_size < 0:
    next_cache_start = 0
  elif required_cache_size == 0:
    next_cache_start = xs.size(1)
  else:
    if xs.size(1) - required_cache_size < 0:
      next_cache_start = 0
    else:
      next_cache_start = xs.size(1) - required_cache_size
  return torch.tensor(next_cache_start, dtype=torch.int64)
