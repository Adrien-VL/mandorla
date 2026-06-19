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
  match args.dataset:
    case "enwik8":
      from mandorla.train.enwik8 import TrainConfig
      from mandorla.train.enwik8 import main as run
    case "enwik8-iter":
      from mandorla.train.enwik8_iter import TrainConfig
      from mandorla.train.enwik8_iter import main as run
    case "enwik8-iter-curriculum":
      from mandorla.train.enwik8_iter_curriculum import TrainConfig
      from mandorla.train.enwik8_iter_curriculum import main as run
    case "enwik8-iter-curriculum-large":
      from mandorla.train.enwik8_iter_curriculum_large import TrainConfig
      from mandorla.train.enwik8_iter_curriculum_large import main as run
    case "enwik8-iter-curriculum-loop":
      from mandorla.train.enwik8_iter_curriculum_loop import TrainConfig
      from mandorla.train.enwik8_iter_curriculum_loop import main as run
    case _:
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
  train.add_argument("dataset", choices=[
    "enwik8",
    "enwik8-iter",
    "enwik8-iter-curriculum",
    "enwik8-iter-curriculum-large",
    "enwik8-iter-curriculum-loop"
  ], help="which variant to train")
  train.add_argument("--total-steps", type=int, help="total training steps (flat only)")
  train.add_argument("--steps-per-phase", type=int, help="steps per layer (iter only)")
  train.add_argument("--batch-size", type=int, help="batch size")
  train.add_argument("--lr", type=float, help="peak learning rate")
  train.add_argument("--seed", type=int, help="random seed")
  train.add_argument("--resume", type=str, help="path to checkpoint to resume from")
  train.add_argument("--out-dir", type=str, help="where to save checkpoints/dashboard")

  return parser


def main() -> None:
  args = build_parser().parse_args()
  match args.command:
    case "info":
      cmd_info()
    case "train":
      cmd_train(args)


if __name__ == "__main__":
  main()