from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class FeedForwardArgs:
  d_model: int
  d_inner: int


class FeedForward(nn.Module):
  def __init__(self, args: FeedForwardArgs):
    super().__init__()
    self.d_model, self.d_inner = args.d_model, args.d_inner

    self.gate_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
    self.up_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
    self.down_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

  def forward(self, x: Tensor) -> Tensor:
    gated, upped = self.gate_proj(x), self.up_proj(x)
    out = F.silu(gated) * upped

    return self.down_proj(out)
