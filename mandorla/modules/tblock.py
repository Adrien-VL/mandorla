from dataclasses import dataclass

from torch import Tensor, nn

from mandorla.modules.attention import Attention, AttentionArgs
from mandorla.modules.ffn import FeedForward, FeedForwardArgs


@dataclass
class TBlockArgs:
  n_heads: int
  d_model: int
  d_head: int
  d_inner: int
  norm_eps: float


class TBlock(nn.Module):
  def __init__(self, args: TBlockArgs):
    super().__init__()
    assert args.d_head % 2 == 0  # RoPE condition
    self.d_model, self.d_head, self.d_inner = args.d_model, args.d_head, args.d_inner
    self.n_heads = args.n_heads
    self.norm_eps = args.norm_eps

    self.attn_norm = nn.RMSNorm(self.d_model, eps=self.norm_eps)
    self.attention = Attention(args=AttentionArgs(
      n_heads=self.n_heads, d_model=self.d_model, d_head=self.d_head,
    ))

    self.ffn_norm = nn.RMSNorm(self.d_model, eps=self.norm_eps)
    self.ffn = FeedForward(args=FeedForwardArgs(
      d_model=self.d_model, d_inner=self.d_inner,
    ))

  def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
    attended = x + self.attention(self.attn_norm(x), freqs_cis)
    out = attended + self.ffn(self.ffn_norm(attended))
    return out