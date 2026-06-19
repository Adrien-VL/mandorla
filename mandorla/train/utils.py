import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mandorla.utils.dashboard import Dashboard


def get_batch(data: Tensor, seq_len: int, batch_size: int, device: str) -> tuple[Tensor, Tensor]:
  ix = torch.randint(0, data.shape[0] - seq_len - 1, (batch_size,))
  x = torch.stack([data[i : i + seq_len] for i in ix])
  y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
  return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: nn.Module, data: Tensor, seq_len: int, batch_size: int,
             device: str, eval_iters: int, depth: int | None = None) -> tuple[float, float]:
  """Returns (mean, std) of cross-entropy loss across `eval_iters` random batches."""
  model.eval()
  losses = []
  for _ in range(eval_iters):
    x, y = get_batch(data, seq_len, batch_size, device)
    logits = model(x) if depth is None else model(x, depth=depth)
    losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
  model.train()
  t = torch.tensor(losses)
  return t.mean().item(), t.std().item()


@torch.no_grad()
def generate(model: nn.Module, prompt: Tensor, max_new_tokens: int, max_seq_len: int,
             depth: int | None = None) -> Tensor:
  model.eval()
  ids = prompt
  for _ in range(max_new_tokens):
    ids_cond = ids if ids.shape[1] <= max_seq_len else ids[:, -max_seq_len:]
    logits = model(ids_cond) if depth is None else model(ids_cond, depth=depth)
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    ids = torch.cat([ids, next_id], dim=1)
  model.train()
  return ids


def save_checkpoint(state: dict, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(state, path)


def load_checkpoint(path: str, device: str) -> dict[str, Any]:
  """Loads a full checkpoint dict. Caller handles which keys to use."""
  return torch.load(path, map_location=device, weights_only=False)


def cosine_lr(step: int, peak_lr: float, warmup: int, total: int, floor_ratio: float = 0.1) -> float:
  """Linear warmup then cosine decay from peak to floor_ratio * peak."""
  if step < warmup:
    return peak_lr * (step + 1) / warmup
  progress = min((step - warmup) / max(1, total - warmup), 1.0)
  return peak_lr * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def setup_dashboard(out_dir: Path | str, title: str, resume: bool = False) -> Dashboard:
  dash = Dashboard(out_dir, title=title)
  if resume:
    dash.load()
  return dash


def ci_bounds(mean: float, std: float, n: int, z: float = 1.96) -> tuple[float, float]:
  """95% confidence interval (default z=1.96) for the mean given n samples."""
  half = z * std / math.sqrt(max(n, 1))
  return mean - half, mean + half