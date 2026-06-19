import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
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


@dataclass
class TrainConfig:
  cache_dir: str = "./data"
  out_dir: str = "./data/base"
  resume: str | None = None

  n_layers: int = 6
  n_heads: int = 8
  d_model: int = 384
  d_head: int = 48
  d_inner: int = 1024
  norm_eps: float = 1e-5
  vocab_size: int = 256
  max_seq_len: int = 512

  batch_size: int = 32
  seq_len: int = 512
  lr: float = 6e-4
  weight_decay: float = 0.1
  beta1: float = 0.9
  beta2: float = 0.95
  grad_clip: float = 1.0
  warmup_steps: int = 300
  total_steps: int = 8_000

  print_every: int = 100
  eval_every: int = 500
  eval_iters: int = 50
  sample_tokens: int = 256
  prompt_bytes: int = 64

  device: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
  )
  seed: int = 42


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

  optim = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr,
    betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
  )

  out_dir = Path(cfg.out_dir)
  start_step = 0
  best_loss = float("inf")

  if cfg.resume:
    ckpt = load_checkpoint(cfg.resume, cfg.device)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"] + 1
    best_loss = ckpt.get("best_loss", ckpt.get("val_loss", float("inf")))
    print(f"resuming from step {ckpt['step']} | best val loss {best_loss:.4f}")

  dashboard = setup_dashboard(out_dir, f"enwik8 AdamW ({n_params/1e6:.1f}M)", resume=bool(cfg.resume))

  model.train()
  t0 = time.time()
  for step in range(start_step, cfg.total_steps):
    lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, cfg.total_steps)
    for g in optim.param_groups:
      g["lr"] = lr

    x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, cfg.device)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optim.step()

    if step % cfg.print_every == 0:
      tps = max(1, step + 1 - start_step) * cfg.batch_size * cfg.seq_len / (time.time() - t0)
      print(f"step {step:>6} | loss {loss.item():.4f} | lr {lr:.2e} | {tps:>7.0f} tok/s")
      dashboard.log_train(step, loss.item(), lr, tps)
      dashboard.flush()

    if step > 0 and step % cfg.eval_every == 0:
      val_mean, val_std = evaluate(model, val_data, cfg.seq_len, cfg.batch_size, cfg.device, cfg.eval_iters)
      val_lo, val_hi = ci_bounds(val_mean, val_std, cfg.eval_iters)
      bpb = val_mean / math.log(2)
      bpb_lo, bpb_hi = val_lo / math.log(2), val_hi / math.log(2)
      print(f"  >> step {step:>6} | val {val_mean:.4f} ±{(val_hi-val_mean):.4f} | bpb {bpb:.4f}")

      prompt = val_data[: cfg.prompt_bytes].unsqueeze(0).to(cfg.device)
      generated = generate(model, prompt, cfg.sample_tokens, cfg.max_seq_len)
      text = bytes(generated[0].tolist()).decode("utf-8", errors="replace")
      print(f"  >> sample:\n{text}\n")

      improved = val_mean < best_loss
      best_loss = min(best_loss, val_mean)

      dashboard.log_eval(step, val_mean, val_lo, val_hi, bpb, bpb_lo, bpb_hi, sample=text)
      dashboard.flush()

      state = {
        "step": step, "model": model.state_dict(), "optim": optim.state_dict(),
        "cfg": asdict(cfg), "val_loss": val_mean, "best_loss": best_loss,
      }
      save_checkpoint(state, out_dir / "latest.pt")
      if improved:
        save_checkpoint(state, out_dir / "best.pt")