import argparse
from dataclasses import fields, replace

import torch

from mandorla import __version__


def cmd_info() -> None:
  print(f"mandorla {__version__}")
  print(f"torch {torch.__version__}")
  if torch.cuda.is_available():
    print(f"cuda: {torch.cuda.get_device_name(0)}")
  elif torch.backends.mps.is_available():
    print("backend: mps")
  else:
    print("backend: cpu")


def cmd_train(args: argparse.Namespace) -> None:
  if args.dataset == "enwik8":
    from mandorla.train.enwik8 import TrainConfig
    from mandorla.train.enwik8 import main as run
  elif args.dataset == "enwik8-sparsify":
    from mandorla.train.sparsify import TrainConfig
    from mandorla.train.sparsify import main as run
  else:
    raise ValueError(f"unknown dataset: {args.dataset}")

  cfg = TrainConfig()
  field_names = {f.name for f in fields(cfg)}
  overrides = {k: v for k, v in vars(args).items() if v is not None and k in field_names}
  run(replace(cfg, **overrides))


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="mandorla")
  sub = parser.add_subparsers(dest="command", required=True)

  sub.add_parser("info", help="print environment info")

  train = sub.add_parser("train", help="train a model")
  train.add_argument("dataset", choices=["enwik8", "enwik8-sparsify"])

  # Mode toggles (boolean: --use-iter / --no-use-iter, etc.)
  for flag in ["use-iter", "use-curriculum", "use-loops", "use-bucket-reweighting", "use-pcgrad"]:
    train.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None,
                       dest=flag.replace("-", "_"))

  # Model
  train.add_argument("--n-layers", type=int, dest="n_layers")
  train.add_argument("--n-heads", type=int, dest="n_heads")
  train.add_argument("--d-model", type=int, dest="d_model")
  train.add_argument("--d-head", type=int, dest="d_head")
  train.add_argument("--d-inner", type=int, dest="d_inner")
  train.add_argument("--max-seq-len", type=int, dest="max_seq_len")

  # Training
  train.add_argument("--steps-per-phase", type=int, dest="steps_per_phase")
  train.add_argument("--batch-size", type=int, dest="batch_size")
  train.add_argument("--seq-len", type=int, dest="seq_len")
  train.add_argument("--lr", type=float)
  train.add_argument("--weight-decay", type=float, dest="weight_decay")
  train.add_argument("--seed", type=int)
  train.add_argument("--resume", type=str)
  train.add_argument("--out-dir", type=str, dest="out_dir")

  # Looping
  train.add_argument("--max-loops", type=int, dest="max_loops")
  train.add_argument("--first-loop-weight", type=float, dest="first_loop_weight")
  train.add_argument("--lambda-mono", type=float, dest="lambda_mono")

  # Sparsify
  train.add_argument("--checkpoint", type=str)
  train.add_argument("--schedule", type=str, choices=["uniform", "ascending", "descending"])
  train.add_argument("--sparsity-max", type=float, dest="sparsity_max")

  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.command == "info":
    cmd_info()
  elif args.command == "train":
    cmd_train(args)


if __name__ == "__main__":
  main()