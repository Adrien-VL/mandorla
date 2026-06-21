import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def _chunk_data(data: Tensor, seq_len: int) -> list[bytes]:
  """Slice data into seq_len-byte chunks for scoring."""
  n = len(data) // seq_len
  return [bytes(data[i * seq_len : (i + 1) * seq_len].tolist()) for i in range(n)]


def _perplexity_loss(chunks: list[bytes], device: str) -> np.ndarray:
  """Per-chunk cross-entropy loss under GPT-2. Lower = more 'common' text."""
  model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
  tok = GPT2TokenizerFast.from_pretrained("gpt2")

  out = np.zeros(len(chunks))
  with torch.no_grad():
    for i, chunk in enumerate(chunks):
      text = chunk.decode("utf-8", errors="replace")
      inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
      if inp.input_ids.shape[1] < 2:
        out[i] = float("inf")
        continue
      out[i] = model(**inp, labels=inp.input_ids).loss.item()
      if i and i % 1000 == 0:
        print(f"  perplexity: {i}/{len(chunks)}")
  return out


def _centroid_distances(chunks: list[bytes], device: str) -> np.ndarray:
  """Per-chunk Euclidean distance from the mean MiniLM embedding."""
  enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
  texts = [c.decode("utf-8", errors="replace") for c in chunks]
  emb = enc.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
  return np.linalg.norm(emb - emb.mean(axis=0), axis=1)


def _normalize(x: np.ndarray) -> np.ndarray:
  """Map finite values to [0, 1]; push non-finite to 1.0 (hardest)."""
  valid = np.isfinite(x)
  x = x.copy()
  if valid.any():
    lo, hi = x[valid].min(), x[valid].max()
    x[valid] = (x[valid] - lo) / max(hi - lo, 1e-10)
  x[~valid] = 1.0
  return x


def _retrieve_cache(
  seq_len: int,
  cache_dir: str | Path,
  name: str,
  alpha: float,
  force: bool
) -> tuple[Path, Any | None]:
  """Retrieve the cached curriculum ranking if present."""
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  path = cache_dir / f"{name}_seq{seq_len}_a{alpha:.2f}.npy"
  if path.exists() and not force:
    return path, np.load(path)
  return path, None


def build_curriculum(
  data: Tensor,
  seq_len: int,
  cache_dir: str | Path,
  name: str = "curriculum",
  alpha: float = 0.7,
  device: str = "mps",
  force: bool = False,
) -> np.ndarray:
  """Compute (and cache) ascending-order chunk indices: easiest first.

  alpha mixes perplexity (1.0) vs centroid distance (0.0). 0.7 is a sensible
  default that emphasizes 'generality under a text prior' with a geometric
  tiebreaker.
  """
  path, cache = _retrieve_cache(seq_len, cache_dir, name, alpha, force)
  if cache is not None:
    return cache

  chunks = _chunk_data(data, seq_len)
  print(f"scoring {len(chunks):,} chunks of {seq_len} bytes")

  print("step 1/2: GPT-2 perplexity")
  ppl = _perplexity_loss(chunks, device)

  print("step 2/2: MiniLM centroid distance")
  dist = _centroid_distances(chunks, device)

  score = alpha * _normalize(ppl) + (1 - alpha) * _normalize(dist)
  sorted_idx = np.argsort(score).astype(np.int64)

  np.save(path, sorted_idx)
  print(f"cached to {path}")
  return sorted_idx


def phase_chunks(sorted_indices, phase, n_phases):
  chunks_needed = math.ceil((phase + 1) * len(sorted_indices) / n_phases)
  return sorted_indices[:chunks_needed]


def get_batch(
  data: Tensor,
  available_chunks: np.ndarray,
  seq_len: int,
  batch_size: int,
  device: str,
  return_ids: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, np.ndarray]:
  """Sample a batch from the chunks unlocked at the current phase.
  If return_ids=True, also returns the chunk indices used (for bucket assignment)."""
  ids = available_chunks[np.random.randint(0, len(available_chunks), size=batch_size)]
  starts = torch.from_numpy(ids).long() * seq_len
  x = torch.stack([data[s : s + seq_len] for s in starts])
  y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
  x = x.to(device, non_blocking=True)
  y = y.to(device, non_blocking=True)
  if return_ids:
    return x, y, ids
  return x, y