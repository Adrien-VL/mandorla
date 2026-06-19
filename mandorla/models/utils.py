import torch
from torch import Tensor


def precompute_freqs_cis(d_head: int, max_seq_len: int, base: float = 10000.0) -> Tensor:
  freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
  t = torch.arange(max_seq_len)
  angles = torch.outer(t, freqs)
  return torch.polar(torch.ones_like(angles), angles)