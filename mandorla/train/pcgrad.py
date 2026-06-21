import random

import torch
from torch import nn


def pcgrad_aggregate(
  losses: list[torch.Tensor], params: list[nn.Parameter], weights: list[float],
) -> list[torch.Tensor]:
  """PCGrad-projected sum of per-task gradients across multiple losses.

  Each loss may live on its own computation graph; retain_graph=True ensures
  downstream code (e.g. monotonic penalty) can still backward through them
  after this function returns.

  Returns a list of gradient tensors aligned with `params`. Caller is
  responsible for setting `p.grad = result[i]` before calling optim.step().
  """
  n = len(losses)
  grads: list[list[torch.Tensor]] = []
  for i, loss in enumerate(losses):
    gi = torch.autograd.grad(
      weights[i] * loss, params, retain_graph=True, allow_unused=True,
    )
    grads.append([
      g if g is not None else torch.zeros_like(p)
      for g, p in zip(gi, params, strict=False)
    ])

  projected = [list(g) for g in grads]
  for i in range(n):
    order = [j for j in range(n) if j != i]
    random.shuffle(order)
    for j in order:
      dot = sum((projected[i][k] * grads[j][k]).sum() for k in range(len(params)))
      if dot.item() < 0:
        norm_sq = sum((grads[j][k] * grads[j][k]).sum() for k in range(len(params)))
        if norm_sq.item() > 1e-12:
          coef = (dot / norm_sq).item()
          for k in range(len(params)):
            projected[i][k] = projected[i][k] - coef * grads[j][k]

  out = [projected[0][k].clone() for k in range(len(params))]
  for i in range(1, n):
    for k in range(len(params)):
      out[k] = out[k] + projected[i][k]
  return out