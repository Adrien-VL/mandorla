import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mandorla.models.transformer import Transformer

# ===== Active params and weight-change tracking =====

def active_params(model: Transformer, depth: int) -> list[nn.Parameter]:
  """Embedding + first `depth` layers + final norm + lm_head."""
  params = list(model.tok_embed.parameters())
  for layer in model.layers[:depth]:
    params.extend(layer.parameters())
  params.extend(model.final_norm.parameters())
  params.extend(model.lm_head.parameters())
  return params


def snapshot_layer_weights(model: Transformer, depth: int) -> list[list[torch.Tensor]]:
  """Detached clones of each layer's parameters, for layers 0..depth-1."""
  return [
    [p.detach().clone() for p in model.layers[i].parameters()]
    for i in range(depth)
  ]


def layer_change_ratios(
  model: Transformer, snapshot: list[list[torch.Tensor]], depth: int,
) -> dict[int, float]:
  """layer_idx -> ||current - snapshot|| / ||snapshot|| per layer."""
  ratios = {}
  for i in range(depth):
    delta_sq, norm_sq = 0.0, 0.0
    for p, p0 in zip(model.layers[i].parameters(), snapshot[i], strict=False):
      delta_sq += (p.detach() - p0).pow(2).sum().item()
      norm_sq += p0.pow(2).sum().item()
    ratios[i] = (delta_sq ** 0.5) / max(norm_sq ** 0.5, 1e-10)
  return ratios


# ===== Looping (horizon-1 BPTT with recompute-from-anchor) =====

def loop_count(phase: int, max_loops: int) -> int:
  """Decreasing loop count by depth. Phase 0 -> max_loops, phase k -> max(1, max_loops - k)."""
  return max(1, max_loops - phase)


def forward_prefix(
  model: Transformer, x: torch.Tensor, depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run embed + layers[0..depth-2] with grad enabled. Returns (h, freqs_cis)."""
  _, d_seq = x.shape
  positions = torch.arange(d_seq, device=x.device)
  freqs_cis = model.freqs_cis[positions]
  h = model.tok_embed(x)
  for layer in model.layers[:depth - 1]:
    h = layer(h, freqs_cis)
  return h, freqs_cis


def compute_loop_per_sample_losses(
  model: Transformer, x: torch.Tensor, y: torch.Tensor, depth: int, n_loops: int,
  vocab_size: int,
) -> list[torch.Tensor]:
  """Per-loop per-sample losses via horizon-1 BPTT with recompute-from-anchor.

  - Loss 0: full grad through embed + earlier layers + 1 application of current layer.
    (The deployable 1-loop path; earlier layers learn from this loss.)
  - Loss i>=1: grad through 2 applications of current layer only, starting from
    a detached anchor at step i-1. Earlier layers are not updated by these losses.
  """
  B, T = x.shape
  h_input, freqs_cis = forward_prefix(model, x, depth)
  current = model.layers[depth - 1]

  with torch.no_grad():
    anchors = [h_input.detach()]
    s = h_input.detach()
    for _ in range(n_loops):
      s = current(s, freqs_cis)
      anchors.append(s)

  per_loop: list[torch.Tensor] = []
  for i in range(n_loops):
    if i == 0:
      s_i = current(h_input, freqs_cis)
    else:
      s_prev = current(anchors[i - 1], freqs_cis)
      s_i = current(s_prev, freqs_cis)
    logits = model.lm_head(model.final_norm(s_i))
    per_sample = F.cross_entropy(
      logits.reshape(-1, vocab_size), y.reshape(-1), reduction='none',
    ).reshape(B, T).mean(dim=1)
    per_loop.append(per_sample)
  return per_loop


# ===== Bucket reweighting =====

def compute_unlock_phases(sorted_idx: np.ndarray, n_phases: int) -> np.ndarray:
  """For each chunk index, the phase at which it first becomes available
  under linear phase_chunks unlocking."""
  n = len(sorted_idx)
  rank = np.empty(n, dtype=np.int64)
  rank[sorted_idx] = np.arange(n, dtype=np.int64)
  return (rank * n_phases) // n


def assign_buckets(
  chunk_ids: np.ndarray, unlock_phases: np.ndarray,
  current_phase: int, recent_window: int,
) -> np.ndarray:
  """Per-sample bucket id: 0 = new (unlocked this phase), 1 = recent
  (within recent_window phases), 2 = old."""
  diff = current_phase - unlock_phases[chunk_ids]
  return np.where(diff == 0, 0, np.where(diff <= recent_window, 1, 2))


def aggregate_buckets(
  per_loop_per_sample: list[torch.Tensor],
  bucket_ids: np.ndarray,
  loop_weights: list[float],
  bucket_weights: list[float],
) -> tuple[list[torch.Tensor], list[float]]:
  """Collapse per-loop per-sample losses into per-bucket scalar losses.
  Returns (losses, weights) for non-empty buckets only, in bucket-id order.
  Each bucket loss = sum over loops of (loop_weight * mean of per-sample loss for samples in bucket).
  """
  device = per_loop_per_sample[0].device
  bucket_ids_t = torch.from_numpy(bucket_ids).to(device)
  losses, weights_out = [], []
  for b_idx, b_w in enumerate(bucket_weights):
    mask = (bucket_ids_t == b_idx)
    if not bool(mask.any()):
      continue
    total = None
    for li, ps in enumerate(per_loop_per_sample):
      contrib = loop_weights[li] * ps[mask].mean()
      total = contrib if total is None else total + contrib
    losses.append(total)
    weights_out.append(b_w)
  return losses, weights_out