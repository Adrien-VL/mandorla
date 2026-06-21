import torch

from mandorla.models.transformer import Transformer


def per_layer_sparsity(
  overall: float, n_layers: int, schedule: str, delta: float,
) -> list[float]:
  """Compute per-layer sparsity targets given an overall (mean) sparsity.
  - uniform: all layers get `overall`.
  - ascending: linearly from overall-delta (layer 0) to overall+delta (last layer).
  - descending: reverse of ascending.
  """
  if schedule == "uniform" or n_layers < 2:
    return [overall] * n_layers
  if schedule == "ascending":
    return [
      max(0.0, min(1.0, overall - delta + 2 * delta * i / (n_layers - 1)))
      for i in range(n_layers)
    ]
  if schedule == "descending":
    return [
      max(0.0, min(1.0, overall + delta - 2 * delta * i / (n_layers - 1)))
      for i in range(n_layers)
    ]
  raise ValueError(f"unknown schedule: {schedule}")


def update_masks(
  model: Transformer, target_sparsities: list[float], masks: dict,
  projections: tuple[str, ...],
) -> dict[tuple[int, str], float]:
  """In-place: apply magnitude pruning to reach target_sparsities per layer.
  Monotonic: intersects with existing masks so a weight, once pruned, stays pruned.
  Returns actual achieved sparsity per (layer_idx, projection_name).
  """
  actual = {}
  for i, layer in enumerate(model.layers):
    s = target_sparsities[i]
    if s <= 0:
      continue
    for name in projections:
      module = getattr(layer.ffn, name)
      w = module.weight
      n = w.numel()
      k = max(1, int(s * n))
      flat_abs = w.detach().abs().flatten()
      threshold = flat_abs.kthvalue(k).values
      new_mask = (w.detach().abs() > threshold).to(w.dtype)
      key = (i, name)
      if key in masks:
        new_mask = new_mask * masks[key]
      masks[key] = new_mask
      with torch.no_grad():
        w.mul_(new_mask)
      actual[key] = 1.0 - new_mask.mean().item()
  return actual


def mask_gradients(model: Transformer, masks: dict) -> None:
  """Zero gradient on pruned weights so they stay zero through optim.step()."""
  for (i, name), mask in masks.items():
    p = getattr(model.layers[i].ffn, name).weight
    if p.grad is not None:
      p.grad.mul_(mask)


def reapply_masks(model: Transformer, masks: dict) -> None:
  """Defensive: re-zero pruned weights after optim.step (in case of float drift)."""
  with torch.no_grad():
    for (i, name), mask in masks.items():
      p = getattr(model.layers[i].ffn, name).weight
      p.mul_(mask)