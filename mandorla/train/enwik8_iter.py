import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mandorla.data.enwik8 import load_enwik8
from mandorla.models.transformer import Transformer, TransformerArgs
from mandorla.train.utils import (
  ci_bounds,
  cosine_lr,
  evaluate,
  generate,
  get_batch,
  load_checkpoint,
  save_checkpoint,
  setup_dashboard,
)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class TrainConfig:
  cache_dir: str = "./data"
  out_dir: str = "./data/iter-deep"
  resume: str | None = None

  # ~14.8M params, 14 layers (deep + thin)
  n_layers: int = 14
  n_heads: int = 8
  d_model: int = 256
  d_head: int = 32
  d_inner: int = 1024
  norm_eps: float = 1e-5
  vocab_size: int = 256
  max_seq_len: int = 512

  steps_per_phase: int = 2500
  warmup_steps: int = 300
  lr: float = 5e-4
  weight_decay: float = 0.1
  beta1: float = 0.9
  beta2: float = 0.95
  grad_clip: float = 1.0
  batch_size: int = 32
  seq_len: int = 512
  forward_dtype: str = "bf16"

  print_every: int = 100
  eval_every: int = 500
  eval_iters: int = 100
  sample_tokens: int = 256
  prompt_bytes: int = 64

  device: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
  )
  seed: int = 42


def _snapshot_layer_weights(model: Transformer, depth: int) -> list[list[torch.Tensor]]:
  """Detached clones of each layer's parameters, for layers 0..depth-1."""
  return [
    [p.detach().clone() for p in model.layers[i].parameters()]
    for i in range(depth)
  ]


def _layer_change_ratios(model: Transformer, snapshot: list[list[torch.Tensor]],
                        depth: int) -> dict[int, float]:
  """Returns layer_idx -> ||current - snapshot|| / ||snapshot|| per layer."""
  ratios = {}
  for i in range(depth):
    delta_sq = 0.0
    norm_sq = 0.0
    for p, p0 in zip(model.layers[i].parameters(), snapshot[i], strict=False):
      delta_sq += (p.detach() - p0).pow(2).sum().item()
      norm_sq += p0.pow(2).sum().item()
    ratios[i] = (delta_sq ** 0.5) / max(norm_sq ** 0.5, 1e-10)
  return ratios


def _active_params(model: Transformer, depth: int) -> list[nn.Parameter]:
  """Embedding + first `depth` layers + final norm + lm_head."""
  params = list(model.tok_embed.parameters())
  for layer in model.layers[:depth]:
    params.extend(layer.parameters())
  params.extend(model.final_norm.parameters())
  params.extend(model.lm_head.parameters())
  return params


def main(cfg: TrainConfig) -> None:
  torch.manual_seed(cfg.seed)

  train_data, val_data, _ = load_enwik8(cfg.cache_dir)
  print(f"train: {len(train_data):,} bytes | val: {len(val_data):,} bytes")

  model = Transformer(TransformerArgs(
    n_layers=cfg.n_layers, n_heads=cfg.n_heads,
    d_model=cfg.d_model, d_head=cfg.d_head, d_inner=cfg.d_inner,
    norm_eps=cfg.norm_eps, vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
  )).to(cfg.device)

  n_params = sum(p.numel() for p in model.parameters())
  print(f"model: {n_params / 1e6:.2f}M params on {cfg.device}")

  out_dir = Path(cfg.out_dir)
  start_phase = 0
  start_step_in_phase = 0
  best_loss = float("inf")

  if cfg.resume:
    ckpt = load_checkpoint(cfg.resume, cfg.device)
    model.load_state_dict(ckpt["model"])
    start_phase = ckpt.get("phase", 0)
    start_step_in_phase = ckpt.get("step_in_phase", 0) + 1
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"resuming from phase {start_phase}, step {start_step_in_phase} | best val {best_loss:.4f}")

  dashboard = setup_dashboard(out_dir, f"enwik8 iter ({n_params/1e6:.1f}M)", resume=bool(cfg.resume))

  global_step = start_phase * cfg.steps_per_phase + start_step_in_phase
  t0 = time.time()
  t_start_step = global_step

  for phase in range(start_phase, cfg.n_layers):
    depth = phase + 1
    active = _active_params(model, depth)
    n_active = sum(p.numel() for p in active)

    optim = torch.optim.AdamW(
      active, lr=cfg.lr,
      betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )

    print(f"\n=== phase {phase} | depth={depth} | {n_active/1e6:.2f}M trainable ===")

    phase_snapshot = _snapshot_layer_weights(model, depth)
    phase_start = start_step_in_phase if phase == start_phase else 0

    model.train()
    for sp in range(phase_start, cfg.steps_per_phase):
      lr = cosine_lr(sp, cfg.lr, cfg.warmup_steps, cfg.steps_per_phase)
      for g in optim.param_groups:
        g["lr"] = lr

      x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, cfg.device)
      dtype = _DTYPES[cfg.forward_dtype]
      with torch.autocast(device_type=cfg.device, dtype=dtype, enabled=(dtype != torch.float32)):
        logits = model(x, depth=depth)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

      optim.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(active, cfg.grad_clip)
      optim.step()

      if sp % cfg.print_every == 0:
        tps = max(1, global_step + 1 - t_start_step) * cfg.batch_size * cfg.seq_len / (time.time() - t0)
        print(f"phase {phase} step {sp:>5} (global {global_step:>6}) | loss {loss.item():.4f} | "
              f"lr {lr:.2e} | {tps:>7.0f} tok/s")
        dashboard.log_train(global_step, loss.item(), lr, tps)
        dashboard.flush()

      if sp > 0 and sp % cfg.eval_every == 0:
        val_mean, val_std = evaluate(
          model, val_data, cfg.seq_len, cfg.batch_size, cfg.device, cfg.eval_iters,
          depth=depth,
        )
        val_lo, val_hi = ci_bounds(val_mean, val_std, cfg.eval_iters)
        bpb = val_mean / math.log(2)
        bpb_lo, bpb_hi = val_lo / math.log(2), val_hi / math.log(2)
        print(f"  >> phase {phase} step {sp} | val {val_mean:.4f} ±{(val_hi-val_mean):.4f} | bpb {bpb:.4f}")

        prompt = val_data[: cfg.prompt_bytes].unsqueeze(0).to(cfg.device)
        generated = generate(
          model, prompt, cfg.sample_tokens, cfg.max_seq_len, depth=depth,
        )
        text = bytes(generated[0].tolist()).decode("utf-8", errors="replace")
        print(f"  >> sample:\n{text}\n")

        improved = val_mean < best_loss
        best_loss = min(best_loss, val_mean)

        dashboard.log_eval(global_step, val_mean, val_lo, val_hi, bpb, bpb_lo, bpb_hi, sample=text)
        layer_changes = _layer_change_ratios(model, phase_snapshot, depth)
        dashboard.log_layer_changes(global_step, layer_changes)
        dashboard.flush()

        state = {
          "phase": phase, "step_in_phase": sp, "model": model.state_dict(),
          "cfg": asdict(cfg), "val_loss": val_mean, "best_loss": best_loss,
        }
        save_checkpoint(state, out_dir / "latest.pt")
        if improved:
          save_checkpoint(state, out_dir / "best.pt")

      global_step += 1

    start_step_in_phase = 0