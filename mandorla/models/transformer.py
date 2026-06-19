from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mandorla.models.utils import precompute_freqs_cis
from mandorla.modules.tblock import TBlock, TBlockArgs


@dataclass
class TransformerArgs:
  n_layers: int
  n_heads: int
  d_model: int
  d_head: int
  d_inner: int
  norm_eps: float
  vocab_size: int
  max_seq_len: int


class Transformer(nn.Module):
  def __init__(self, args: TransformerArgs):
    super().__init__()
    self.d_model, self.d_head, self.d_inner = args.d_model, args.d_head, args.d_inner
    self.n_layers, self.n_heads = args.n_layers, args.n_heads
    self.norm_eps = args.norm_eps
    self.vocab_size = args.vocab_size
    self.max_seq_len = args.max_seq_len

    self.tok_embed = nn.Embedding(self.vocab_size, self.d_model)

    self.layers = nn.ModuleList([TBlock(TBlockArgs(
      n_heads=self.n_heads,
      d_model=self.d_model,
      d_head=self.d_head,
      d_inner=self.d_inner,
      norm_eps=self.norm_eps,
    )) for _ in range(self.n_layers)])

    self.final_norm = nn.RMSNorm(self.d_model, eps=self.norm_eps)
    self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    self.register_buffer(
      "freqs_cis",
      precompute_freqs_cis(self.d_head, self.max_seq_len),
      persistent=False,
    )

    self._init_weights()

  def _init_weights(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, tokens: Tensor, positions: Tensor | None = None,
              depth: int | None = None) -> Tensor:
    _, d_seq = tokens.shape

    if depth is None:
      depth = self.n_layers
    if positions is None:
      positions = torch.arange(d_seq, device=tokens.device)

    freqs_cis = self.freqs_cis[positions]

    h = self.tok_embed(tokens)
    for layer in self.layers[:depth]:
      h = layer(h, freqs_cis)
    h = self.final_norm(h)
    return self.lm_head(h)