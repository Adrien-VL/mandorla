import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def _download(cache_dir: Path) -> Path:
  cache_dir.mkdir(parents=True, exist_ok=True)
  raw_path = cache_dir / "enwik8"
  if raw_path.exists():
    return raw_path

  zip_path = cache_dir / "enwik8.zip"
  if not zip_path.exists():
    print(f"downloading enwik8 from {ENWIK8_URL}")
    urllib.request.urlretrieve(ENWIK8_URL, zip_path)
  with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(cache_dir)
  return raw_path


def load_enwik8(cache_dir: Path | str = "./data") -> tuple[Tensor, Tensor, Tensor]:
  """Returns (train, val, test) as 1D long tensors of byte values. Standard 90/5/5 split."""
  raw_path = _download(Path(cache_dir))
  arr = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8).copy()
  data = torch.from_numpy(arr).long()
  n_train, n_val = 90_000_000, 5_000_000
  return data[:n_train], data[n_train : n_train + n_val], data[n_train + n_val :]