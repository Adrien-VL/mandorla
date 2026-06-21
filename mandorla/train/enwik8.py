import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mandorla.data import curriculum
from mandorla.data.enwik8 import load_enwik8
from mandorla.models.transformer import Transformer, TransformerArgs
from mandorla.train.iter_methods import (
  active_params,
  aggregate_buckets,
  assign_buckets,
  compute_loop_per_sample_losses,
  compute_unlock_phases,
  layer_change_ratios,
  loop_count,
  snapshot_layer_weights,
)
from mandorla.train.pcgrad import pcgrad_aggregate
from mandorla.train.utils import (
  ci_bounds,
  cosine_lr,
  evaluate,
  generate,
  load_checkpoint,
  save_checkpoint,
  setup_dashboard,
)
from mandorla.train.utils import (
  get_batch as get_batch_random,
)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class TrainConfig:
  # ===== Mode toggles =====
  use_iter: bool = False
  use_curriculum: bool = False
  use_loops: bool = False
  use_bucket_reweighting: bool = False
  use_pcgrad: bool = True

  # ===== Model =====
  n_layers: int = 14
  n_heads: int = 8
  d_model: int = 256
  d_head: int = 32
  d_inner: int = 1024
  norm_eps: float = 1e-5
  vocab_size: int = 256
  max_seq_len: int = 512

  # ===== Training =====
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

  # ===== Curriculum =====
  curriculum_alpha: float = 0.7

  # ===== Looping =====
  max_loops: int = 4
  first_loop_weight: float = 2.0
  lambda_mono: float = 0.0

  # ===== Bucket reweighting =====
  bucket_weight_new: float = 0.5
  bucket_weight_recent: float = 0.25
  bucket_weight_old: float = 0.25
  recent_window: int = 2

  # ===== I/O =====
  cache_dir: str = "./data"
  out_dir: str | None = None
  resume: str | None = None
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


def validate_config(cfg: TrainConfig) -> None:
  if cfg.use_curriculum and not cfg.use_iter:
    raise ValueError("use_curriculum requires use_iter")
  if cfg.use_loops and not cfg.use_iter:
    raise ValueError("use_loops requires use_iter")
  if cfg.use_bucket_reweighting and not cfg.use_curriculum:
    raise ValueError("use_bucket_reweighting requires use_curriculum")


def make_out_dir(cfg: TrainConfig) -> str:
  parts = ["enwik8"]
  if not cfg.use_iter:
    parts.append("flat")
  else:
    parts.append("iter")
    if cfg.use_curriculum: parts.append("curric")
    if cfg.use_loops: parts.append(f"loop{cfg.max_loops}")
    if cfg.use_bucket_reweighting: parts.append("buckets")
  return f"./data/{'-'.join(parts)}"


def make_title(cfg: TrainConfig, n_params: int) -> str:
  return f"{make_out_dir(cfg).split('/')[-1]} ({n_params / 1e6:.1f}M)"


def main(cfg: TrainConfig) -> None:
  validate_config(cfg)
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  out_dir = Path(cfg.out_dir or make_out_dir(cfg))
  print(f"out_dir: {out_dir}")
  print(f"flags: iter={cfg.use_iter} curriculum={cfg.use_curriculum} "
        f"loops={cfg.use_loops} buckets={cfg.use_bucket_reweighting} pcgrad={cfg.use_pcgrad}")

  train_data, val_data, _ = load_enwik8(cfg.cache_dir)
  print(f"train: {len(train_data):,} bytes | val: {len(val_data):,} bytes")

  sorted_idx, unlock_phases = None, None
  if cfg.use_curriculum:
    sorted_idx = curriculum.build_curriculum(
      train_data, seq_len=cfg.seq_len, cache_dir=cfg.cache_dir,
      alpha=cfg.curriculum_alpha, device=cfg.device,
    )
    print(f"curriculum: {len(sorted_idx):,} chunks sorted (easy-first)")
    if cfg.use_bucket_reweighting:
      unlock_phases = compute_unlock_phases(sorted_idx, cfg.n_layers)
      print(f"bucket reweighting: new={cfg.bucket_weight_new}, "
            f"recent={cfg.bucket_weight_recent} (window={cfg.recent_window}), "
            f"old={cfg.bucket_weight_old}")

  model = Transformer(TransformerArgs(
    n_layers=cfg.n_layers, n_heads=cfg.n_heads,
    d_model=cfg.d_model, d_head=cfg.d_head, d_inner=cfg.d_inner,
    norm_eps=cfg.norm_eps, vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
  )).to(cfg.device)
  n_params = sum(p.numel() for p in model.parameters())
  print(f"model: {n_params / 1e6:.2f}M params on {cfg.device}")

  start_phase, start_step_in_phase = 0, 0
  best_loss = float("inf")
  if cfg.resume:
    ckpt = load_checkpoint(cfg.resume, cfg.device)
    model.load_state_dict(ckpt["model"])
    start_phase = ckpt.get("phase", 0)
    start_step_in_phase = ckpt.get("step_in_phase", 0) + 1
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"resuming from phase {start_phase}, step {start_step_in_phase} | best val {best_loss:.4f}")

  dashboard = setup_dashboard(out_dir, make_title(cfg, n_params), resume=bool(cfg.resume))

  n_phases = cfg.n_layers if cfg.use_iter else 1
  global_step = start_phase * cfg.steps_per_phase + start_step_in_phase
  t0 = time.time()
  t_start_step = global_step

  bucket_weights = [cfg.bucket_weight_new, cfg.bucket_weight_recent, cfg.bucket_weight_old]

  for phase in range(start_phase, n_phases):
    depth = (phase + 1) if cfg.use_iter else cfg.n_layers
    n_loops = loop_count(phase, cfg.max_loops) if cfg.use_loops else 1
    active = active_params(model, depth) if cfg.use_iter else list(model.parameters())
    n_active = sum(p.numel() for p in active)

    if cfg.use_curriculum:
      available = np.asarray(curriculum.phase_chunks(sorted_idx, phase, n_phases))
      chunk_info = f"{len(available):,}/{len(sorted_idx):,} chunks ({100*len(available)/len(sorted_idx):.1f}%)"
    else:
      available, chunk_info = None, "full data"

    optim = torch.optim.AdamW(
      active, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )

    print(f"\n=== phase {phase} | depth={depth} | n_loops={n_loops} | "
          f"{n_active/1e6:.2f}M trainable | {chunk_info} ===")

    phase_snapshot = snapshot_layer_weights(model, depth) if cfg.use_iter else None
    phase_start = start_step_in_phase if phase == start_phase else 0
    loop_weights = [cfg.first_loop_weight] + [1.0] * (n_loops - 1) if n_loops > 1 else [1.0]

    model.train()
    for sp in range(phase_start, cfg.steps_per_phase):
      lr = cosine_lr(sp, cfg.lr, cfg.warmup_steps, cfg.steps_per_phase)
      for g in optim.param_groups:
        g["lr"] = lr

      if cfg.use_curriculum:
        x, y, batch_ids = curriculum.get_batch(
          train_data, available, cfg.seq_len, cfg.batch_size, cfg.device, return_ids=True,
        )
      else:
        x, y = get_batch_random(train_data, cfg.seq_len, cfg.batch_size, cfg.device)
        batch_ids = None

      dtype = _DTYPES[cfg.forward_dtype]
      autocast_ctx = torch.autocast(
        device_type=cfg.device, dtype=dtype, enabled=(dtype != torch.float32),
      )

      optim.zero_grad(set_to_none=True)

      with autocast_ctx:
        if n_loops > 1:
          per_loop = compute_loop_per_sample_losses(
            model, x, y, depth, n_loops, cfg.vocab_size,
          )
        else:
          logits = model(x, depth=depth)
          per_sample = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction='none',
          ).reshape(x.shape[0], x.shape[1]).mean(dim=1)
          per_loop = [per_sample]

      if cfg.use_bucket_reweighting and batch_ids is not None:
        bucket_ids = assign_buckets(batch_ids, unlock_phases, phase, cfg.recent_window)
        task_losses, task_weights = aggregate_buckets(
          per_loop, bucket_ids, loop_weights, bucket_weights,
        )
      elif n_loops > 1:
        task_losses = [pl.mean() for pl in per_loop]
        task_weights = loop_weights
      else:
        task_losses = [per_loop[0].mean()]
        task_weights = [1.0]

      if len(task_losses) >= 2 and cfg.use_pcgrad:
        aggregated = pcgrad_aggregate(task_losses, active, task_weights)
        for p, g in zip(active, aggregated, strict=False):
          p.grad = g
      else:
        total = sum(w * l for w, l in zip(task_weights, task_losses, strict=False))
        total.backward(retain_graph=(cfg.lambda_mono > 0 and n_loops > 1))

      if cfg.lambda_mono > 0 and n_loops > 1:
        loop_scalar = [l.mean() for l in per_loop]
        mono = sum(F.relu(loop_scalar[i + 1] - loop_scalar[i]) for i in range(n_loops - 1))
        mono_grads = torch.autograd.grad(cfg.lambda_mono * mono, active, allow_unused=True)
        for p, mg in zip(active, mono_grads, strict=False):
          if mg is not None:
            p.grad = mg if p.grad is None else (p.grad + mg)

      loss_value = per_loop[0].mean().item()
      torch.nn.utils.clip_grad_norm_(active, cfg.grad_clip)
      optim.step()

      if sp % cfg.print_every == 0:
        tps = max(1, global_step + 1 - t_start_step) * cfg.batch_size * cfg.seq_len / (time.time() - t0)
        print(f"phase {phase} step {sp:>5} (global {global_step:>6}) | loss {loss_value:.4f} | "
              f"loops {n_loops} | tasks {len(task_losses)} | lr {lr:.2e} | {tps:>7.0f} tok/s")
        dashboard.log_train(global_step, loss_value, lr, tps)
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
        generated = generate(model, prompt, cfg.sample_tokens, cfg.max_seq_len, depth=depth)
        text = bytes(generated[0].tolist()).decode("utf-8", errors="replace")
        print(f"  >> sample:\n{text}\n")

        improved = val_mean < best_loss
        best_loss = min(best_loss, val_mean)

        dashboard.log_eval(global_step, val_mean, val_lo, val_hi, bpb, bpb_lo, bpb_hi, sample=text)
        if phase_snapshot is not None:
          changes = layer_change_ratios(model, phase_snapshot, depth)
          dashboard.log_layer_changes(global_step, changes)
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