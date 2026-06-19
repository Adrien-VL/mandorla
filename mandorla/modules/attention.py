from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
  _, _, d_seq, _ = x.shape

  # split the head dim into pairs and view each pair as a complex number
  x_paired = rearrange(x.float(), "b h s (d p) -> b h s d p", p=2)
  x_complex = torch.view_as_complex(x_paired)

  # broadcast freqs_cis (d_seq, d_head/2) over batch and heads
  freqs_cis = rearrange(freqs_cis[:d_seq], "s d -> 1 1 s d")

  # rotate, then unfold pairs back into the head dim
  x_rotated = torch.view_as_real(x_complex * freqs_cis)
  return rearrange(x_rotated, "b h s d p -> b h s (d p)").type_as(x)

def causal_mask_mod(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
  return q_idx >= kv_idx

def create_causal_block_mask(seq_len: int) -> BlockMask:
  return create_block_mask(causal_mask_mod, None, None, seq_len, seq_len, _compile=True)

flex_attn = torch.compile(flex_attention, dynamic=False)


@dataclass
class AttentionArgs:
  n_heads: int
  d_model: int
  d_head: int


class Attention(nn.Module):
  def __init__(self, args: AttentionArgs):
    super().__init__()
    self.d_model = args.d_model
    self.d_head = args.d_head
    self.n_heads = args.n_heads
    d_inner = self.n_heads * self.d_head

    self.q_proj = nn.Linear(self.d_model, d_inner, bias=False)
    self.k_proj = nn.Linear(self.d_model, d_inner, bias=False)
    self.v_proj = nn.Linear(self.d_model, d_inner, bias=False)
    self.out = nn.Linear(d_inner, self.d_model, bias=False)

    self._block_mask: BlockMask | None = None

  def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
    _, d_seq, _ = x.shape

    q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
    k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
    v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_heads)

    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if self._block_mask is None or self._block_mask.shape[-1] != d_seq:
      self._block_mask = create_causal_block_mask(d_seq)

    if x.device.type == "cuda":
      if self._block_mask is None or self._block_mask.shape[-1] != d_seq:
        self._block_mask = create_causal_block_mask(d_seq)
      out = flex_attn(q, k, v, block_mask=self._block_mask)
    else:
      out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    out = rearrange(out, "b h s d -> b s (h d)")

    return self.out(out)
