import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mandorla.data import curriculum
from mandorla.data.enwik8 import load_enwik8
from mandorla.models.transformer import Transformer, TransformerArgs
from mandorla.train.utils import (
  ci_bounds,
  cosine_lr,
  evaluate,
  generate,
  load_checkpoint,
  save_checkpoint,
  setup_dashboard,
)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class TrainConfig:
  cache_dir: str = "./data"
  out_dir: str = "./data/iter-curriculum-loop"
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

  # curriculum
  curriculum_alpha: float = 0.7

  # looping
  max_loops: int = 4         # phase 0 loops this many times; decreases by 1 per phase, floored at 1
  first_loop_weight: float = 2.0   # heavier loss on the deployable (1-loop) output
  use_pcgrad: bool = True
  lambda_mono: float = 0.0   # monotonic penalty weight; 0 = off

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
  return [
    [p.detach().clone() for p in model.layers[i].parameters()]
    for i in range(depth)
  ]


def _layer_change_ratios(model: Transformer, snapshot: list[list[torch.Tensor]],
                        depth: int) -> dict[int, float]:
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
  params = list(model.tok_embed.parameters())
  for layer in model.layers[:depth]:
    params.extend(layer.parameters())
  params.extend(model.final_norm.parameters())
  params.extend(model.lm_head.parameters())
  return params


def loop_count(phase: int, max_loops: int) -> int:
  """Decreasing loop count by depth. Phase 0 -> max_loops, phase k -> max(1, max_loops - k)."""
  return max(1, max_loops - phase)


def _forward_prefix(model: Transformer, x: torch.Tensor, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Run embed + layers[0..depth-2] with grad enabled. Returns (h_input_to_current, freqs_cis)."""
  _, d_seq = x.shape
  positions = torch.arange(d_seq, device=x.device)
  freqs_cis = model.freqs_cis[positions]
  h = model.tok_embed(x)
  for layer in model.layers[:depth - 1]:
    h = layer(h, freqs_cis)
  return h, freqs_cis


def compute_loop_losses(
  model: Transformer, x: torch.Tensor, y: torch.Tensor, depth: int, n_loops: int,
  vocab_size: int,
) -> list[torch.Tensor]:
  """Per-loop losses with horizon-1 BPTT via recompute-from-anchor.

  - Loss 0: full grad through embed + earlier layers + 1 application of current layer.
    (This is the "deployable" 1-loop path; earlier layers learn from this loss.)
  - Loss i>=1: grad through 2 applications of current layer only, starting from
    a detached anchor at step i-1. Earlier layers are not updated by these losses.
  """
  h_input, freqs_cis = _forward_prefix(model, x, depth)
  current = model.layers[depth - 1]

  # Build anchor states with no_grad (cheap; no graph memory)
  with torch.no_grad():
    anchors = [h_input.detach()]
    s = h_input.detach()
    for _ in range(n_loops):
      s = current(s, freqs_cis)
      anchors.append(s)

  losses: list[torch.Tensor] = []
  for i in range(n_loops):
    if i == 0:
      # First loop: replay 1 step from grad-enabled h_input.
      # Grad flows: current layer (this step) + back through embed + earlier layers.
      s_i = current(h_input, freqs_cis)
    else:
      # Replay 2 steps from detached anchor at i-1.
      # Grad flows: current layer at step i AND step i-1 only.
      s_prev = current(anchors[i - 1], freqs_cis)
      s_i = current(s_prev, freqs_cis)
    logits = model.lm_head(model.final_norm(s_i))
    losses.append(F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)))

  return losses


def pcgrad_aggregate(
  losses: list[torch.Tensor], params: list[nn.Parameter], weights: list[float],
) -> list[torch.Tensor]:
  """Compute the PCGrad-projected sum of per-task gradients.

  Returns a list of gradient tensors aligned with `params`. Each loss may live
  on its own computation graph; we use retain_graph=True so subsequent users
  (e.g. monotonic penalty) can still backward through them.
  """
  n = len(losses)

  # Per-task gradients
  grads: list[list[torch.Tensor]] = []
  for i, loss in enumerate(losses):
    gi = torch.autograd.grad(
      weights[i] * loss, params, retain_graph=True, allow_unused=True,
    )
    grads.append([
      g if g is not None else torch.zeros_like(p)
      for g, p in zip(gi, params, strict=False)
    ])

  # PCGrad projection: for each task, project away conflicting components from others.
  projected = [list(g) for g in grads]
  task_indices = list(range(n))
  for i in range(n):
    order = [j for j in task_indices if j != i]
    random.shuffle(order)
    for j in order:
      dot = sum((projected[i][k] * grads[j][k]).sum() for k in range(len(params)))
      if dot.item() < 0:
        norm_sq = sum((grads[j][k] * grads[j][k]).sum() for k in range(len(params)))
        if norm_sq.item() > 1e-12:
          coef = (dot / norm_sq).item()
          for k in range(len(params)):
            projected[i][k] = projected[i][k] - coef * grads[j][k]

  # Sum projected gradients
  out = [projected[0][k].clone() for k in range(len(params))]
  for i in range(1, n):
    for k in range(len(params)):
      out[k] = out[k] + projected[i][k]
  return out


def main(cfg: TrainConfig) -> None:
  torch.manual_seed(cfg.seed)
  random.seed(cfg.seed)

  train_data, val_data, _ = load_enwik8(cfg.cache_dir)
  print(f"train: {len(train_data):,} bytes | val: {len(val_data):,} bytes")

  sorted_idx = curriculum.build_curriculum(
    train_data, seq_len=cfg.seq_len, cache_dir=cfg.cache_dir,
    alpha=cfg.curriculum_alpha, device=cfg.device,
  )
  print(f"curriculum: {len(sorted_idx):,} chunks sorted (easy-first)")

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

  dashboard = setup_dashboard(
    out_dir, f"enwik8 iter+curric+loop ({n_params/1e6:.1f}M)", resume=bool(cfg.resume),
  )

  global_step = start_phase * cfg.steps_per_phase + start_step_in_phase
  t0 = time.time()
  t_start_step = global_step

  for phase in range(start_phase, cfg.n_layers):
    depth = phase + 1
    n_loops = loop_count(phase, cfg.max_loops)
    active = _active_params(model, depth)
    n_active = sum(p.numel() for p in active)

    available = curriculum.phase_chunks(sorted_idx, phase, cfg.n_layers)

    optim = torch.optim.AdamW(
      active, lr=cfg.lr,
      betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )

    print(f"\n=== phase {phase} | depth={depth} | n_loops={n_loops} | "
          f"{n_active/1e6:.2f}M trainable | "
          f"{len(available):,}/{len(sorted_idx):,} chunks "
          f"({100*len(available)/len(sorted_idx):.1f}%) ===")

    phase_snapshot = _snapshot_layer_weights(model, depth)
    phase_start = start_step_in_phase if phase == start_phase else 0

    # Per-loop weights: first loop heaviest (it's what gets deployed)
    weights = [cfg.first_loop_weight] + [1.0] * (n_loops - 1)

    model.train()
    for sp in range(phase_start, cfg.steps_per_phase):
      lr = cosine_lr(sp, cfg.lr, cfg.warmup_steps, cfg.steps_per_phase)
      for g in optim.param_groups:
        g["lr"] = lr

      x, y = curriculum.get_batch(
        train_data, available, cfg.seq_len, cfg.batch_size, cfg.device,
      )

      dtype = _DTYPES[cfg.forward_dtype]
      autocast_ctx = torch.autocast(
        device_type=cfg.device, dtype=dtype, enabled=(dtype != torch.float32),
      )

      optim.zero_grad(set_to_none=True)

      if n_loops == 1:
        # Standard single-pass path — no PCGrad needed.
        with autocast_ctx:
          logits = model(x, depth=depth)
          loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        loss_value = loss.item()
      else:
        # Looped path with horizon-1 BPTT.
        with autocast_ctx:
          losses = compute_loop_losses(model, x, y, depth, n_loops, cfg.vocab_size)

        if cfg.use_pcgrad:
          aggregated = pcgrad_aggregate(losses, active, weights)
          for p, g in zip(active, aggregated, strict=False):
            p.grad = g
        else:
          total = sum(w * l for w, l in zip(weights, losses, strict=False))
          total.backward(retain_graph=(cfg.lambda_mono > 0))

        # Optional: monotonic penalty added on top of PCGrad result.
        if cfg.lambda_mono > 0:
          mono = sum(F.relu(losses[i + 1] - losses[i]) for i in range(n_loops - 1))
          mono_grads = torch.autograd.grad(
            cfg.lambda_mono * mono, active, allow_unused=True,
          )
          for p, mg in zip(active, mono_grads, strict=False):
            if mg is not None:
              if p.grad is None:
                p.grad = mg
              else:
                p.grad = p.grad + mg

        # Report the first-loop loss as the headline number (matches deployment).
        loss_value = losses[0].item()

      torch.nn.utils.clip_grad_norm_(active, cfg.grad_clip)
      optim.step()

      if sp % cfg.print_every == 0:
        tps = max(1, global_step + 1 - t_start_step) * cfg.batch_size * cfg.seq_len / (time.time() - t0)
        print(f"phase {phase} step {sp:>5} (global {global_step:>6}) | loss {loss_value:.4f} | "
              f"loops {n_loops} | lr {lr:.2e} | {tps:>7.0f} tok/s")
        dashboard.log_train(global_step, loss_value, lr, tps)
        dashboard.flush()

      if sp > 0 and sp % cfg.eval_every == 0:
        # Eval uses single-pass forward (the deployable mode for layers past their phase).
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