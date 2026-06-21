import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from mandorla.data.enwik8 import load_enwik8
from mandorla.models.transformer import Transformer, TransformerArgs
from mandorla.train.pruning import (
  mask_gradients,
  per_layer_sparsity,
  reapply_masks,
  update_masks,
)
from mandorla.train.utils import (
  ci_bounds,
  cosine_lr,
  evaluate,
  get_batch,
  load_checkpoint,
  save_checkpoint,
  setup_dashboard,
)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class TrainConfig:
  checkpoint: str = ""
  cache_dir: str = "./data"
  out_dir: str = "./data/sparsify"

  # Sparsity sweep
  sparsity_start: float = 0.1
  sparsity_step: float = 0.1
  sparsity_max: float = 0.8
  schedule: str = "ascending"
  schedule_delta: float = 0.2
  prune_projections: tuple = ("w1", "w2", "w3")

  # Fine-tune between sparsity levels
  finetune_steps: int = 500
  warmup_steps: int = 50
  lr: float = 5e-5
  weight_decay: float = 0.0
  beta1: float = 0.9
  beta2: float = 0.95
  grad_clip: float = 1.0
  batch_size: int = 32
  seq_len: int = 512
  forward_dtype: str = "bf16"

  # Eval / stopping
  eval_iters: int = 100
  print_every: int = 50
  target_degradation: float = 0.10

  device: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
  )
  seed: int = 42


def build_model_from_ckpt_cfg(ckpt_cfg: dict, device: str) -> Transformer:
  args = TransformerArgs(
    n_layers=ckpt_cfg.get("n_layers", 14),
    n_heads=ckpt_cfg.get("n_heads", 8),
    d_model=ckpt_cfg.get("d_model", 256),
    d_head=ckpt_cfg.get("d_head", 32),
    d_inner=ckpt_cfg.get("d_inner", 1024),
    norm_eps=ckpt_cfg.get("norm_eps", 1e-5),
    vocab_size=ckpt_cfg.get("vocab_size", 256),
    max_seq_len=ckpt_cfg.get("max_seq_len", 512),
  )
  return Transformer(args).to(device)


def main(cfg: TrainConfig) -> None:
  torch.manual_seed(cfg.seed)
  if not cfg.checkpoint:
    raise ValueError("--checkpoint is required for sparsify")

  train_data, val_data, _ = load_enwik8(cfg.cache_dir)
  print(f"train: {len(train_data):,} bytes | val: {len(val_data):,} bytes")

  ckpt = load_checkpoint(cfg.checkpoint, cfg.device)
  ckpt_cfg = ckpt.get("cfg", {})
  model = build_model_from_ckpt_cfg(ckpt_cfg, cfg.device)
  model.load_state_dict(ckpt["model"])
  n_params = sum(p.numel() for p in model.parameters())
  print(f"model: {n_params / 1e6:.2f}M params on {cfg.device}")
  print(f"loaded checkpoint: {cfg.checkpoint}")

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  dashboard = setup_dashboard(
    out_dir, f"sparsify ({n_params/1e6:.1f}M, {cfg.schedule})", resume=False,
  )

  n_layers = ckpt_cfg.get("n_layers", len(model.layers))
  masks: dict = {}

  val_mean, val_std = evaluate(
    model, val_data, cfg.seq_len, cfg.batch_size, cfg.device, cfg.eval_iters,
  )
  val_lo, val_hi = ci_bounds(val_mean, val_std, cfg.eval_iters)
  bpb = val_mean / math.log(2)
  print(f"\n=== baseline (sparsity 0.0) | val {val_mean:.4f} ±{val_hi-val_mean:.4f} | bpb {bpb:.4f} ===")
  dashboard.log_eval(0, val_mean, val_lo, val_hi, bpb, val_lo/math.log(2), val_hi/math.log(2),
                     sample="baseline (no pruning)")

  best_val = val_mean
  best_sparsity = 0.0
  global_step = 0
  iteration = 0
  sparsity = cfg.sparsity_start

  while sparsity <= cfg.sparsity_max + 1e-9:
    sparsities = per_layer_sparsity(sparsity, n_layers, cfg.schedule, cfg.schedule_delta)
    actual = update_masks(model, sparsities, masks, cfg.prune_projections)

    avg_actual = sum(actual.values()) / max(len(actual), 1) if actual else 0.0
    print(f"\n=== iter {iteration} | target {sparsity:.2f} | "
          f"avg actual sparsity {avg_actual:.2f} | schedule={cfg.schedule} ===")
    for i, s in enumerate(sparsities):
      keys = [(i, name) for name in cfg.prune_projections if (i, name) in actual]
      if keys:
        layer_actual = sum(actual[k] for k in keys) / len(keys)
        print(f"  layer {i:2d}: target {s:.2f}, actual {layer_actual:.2f}")

    val_pre, _ = evaluate(
      model, val_data, cfg.seq_len, cfg.batch_size, cfg.device, cfg.eval_iters,
    )
    print(f"  pre-finetune val: {val_pre:.4f} (bpb {val_pre/math.log(2):.4f})")

    optim = torch.optim.AdamW(
      model.parameters(), lr=cfg.lr,
      betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )
    model.train()
    t0 = time.time()
    for sp in range(cfg.finetune_steps):
      lr = cosine_lr(sp, cfg.lr, cfg.warmup_steps, cfg.finetune_steps)
      for g in optim.param_groups:
        g["lr"] = lr

      x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, cfg.device)

      dtype = _DTYPES[cfg.forward_dtype]
      with torch.autocast(
        device_type=cfg.device, dtype=dtype, enabled=(dtype != torch.float32),
      ):
        logits = model(x)
        loss = F.cross_entropy(
          logits.reshape(-1, ckpt_cfg.get("vocab_size", 256)), y.reshape(-1),
        )

      optim.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
      mask_gradients(model, masks)
      optim.step()
      reapply_masks(model, masks)

      if sp % cfg.print_every == 0:
        tps = (sp + 1) * cfg.batch_size * cfg.seq_len / (time.time() - t0)
        print(f"  ft step {sp:>4} | loss {loss.item():.4f} | lr {lr:.2e} | {tps:.0f} tok/s")
        dashboard.log_train(global_step, loss.item(), lr, tps)
        dashboard.flush()
      global_step += 1

    val_post, val_std = evaluate(
      model, val_data, cfg.seq_len, cfg.batch_size, cfg.device, cfg.eval_iters,
    )
    val_lo, val_hi = ci_bounds(val_post, val_std, cfg.eval_iters)
    bpb_post = val_post / math.log(2)
    print(f"  post-finetune val: {val_post:.4f} ±{val_hi-val_post:.4f} (bpb {bpb_post:.4f})")
    dashboard.log_eval(
      global_step, val_post, val_lo, val_hi, bpb_post,
      val_lo/math.log(2), val_hi/math.log(2),
      sample=f"target_sparsity={sparsity:.2f}, actual={avg_actual:.2f}",
    )
    dashboard.flush()

    state = {
      "model": model.state_dict(),
      "cfg": asdict(cfg),
      "sparsity": sparsity,
      "actual_sparsity": avg_actual,
      "masks": {f"{k[0]}_{k[1]}": v.cpu() for k, v in masks.items()},
      "val_loss": val_post,
      "bpb": bpb_post,
    }

    if val_post < best_val:
      best_val = val_post
      best_sparsity = sparsity
      save_checkpoint(state, out_dir / "best.pt")
      print(f"  *** new best: val {best_val:.4f} (bpb {bpb_post:.4f}) at sparsity {sparsity:.2f} ***")

    save_checkpoint(state, out_dir / "latest.pt")

    if val_post > best_val + cfg.target_degradation:
      print(f"\n=== STOPPING: val degraded by more than {cfg.target_degradation} "
            f"from best ({val_post:.4f} > {best_val:.4f}) ===")
      break

    sparsity += cfg.sparsity_step
    iteration += 1

  print(f"\nFinal best: val {best_val:.4f} (bpb {best_val/math.log(2):.4f}) "
        f"at overall sparsity {best_sparsity:.2f}")