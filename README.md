# Mandorla

Research project exploring iterative training, data curriculum, intra-layer
looping, and compositional generalization for byte-level transformers on
enwik8. The architectural progression — flat baseline → iter → iter+curriculum
→ iter+curriculum+loops → diamond (planned) — is the core experimental program;
see the accompanying theory document for motivation.

## Setup

```bash
uv sync
uv run python main.py info
```

Python 3.13. Supports CUDA, MPS (Apple Silicon), and CPU. First curriculum
build pulls down GPT-2 and MiniLM weights from HuggingFace.

## Training variants

All variants run through a single entry point with toggle flags. Output
directories are auto-generated from the active flags; override with `--out-dir`.

### 1. Flat baseline

Standard AdamW training with full network depth and random batch sampling.
No curriculum, no iter, no loops.

```bash
uv run python main.py train enwik8
```
→ `./data/enwik8-flat/`

### 2. Iter (depth curriculum)

Layer count grows from 1 to `n_layers` over `n_layers` phases. Each phase
trains for `steps_per_phase` steps. No data curriculum; samples from full
distribution at each phase.

```bash
uv run python main.py train enwik8 --use-iter
```
→ `./data/enwik8-iter/`

### 3. Iter + data curriculum

Adds difficulty-ranked data exposure. Phase k unlocks the easiest
(k+1)/n_phases fraction of chunks. Difficulty scored by GPT-2 perplexity +
MiniLM centroid distance (mixed with `--curriculum-alpha`, default 0.7).

```bash
uv run python main.py train enwik8 --use-iter --use-curriculum
```
→ `./data/enwik8-iter-curric/`

First run takes ~90 minutes to score and cache the curriculum
(`data/curriculum_seq*_a*.npy`). Subsequent runs reuse the cache.

### 4. Iter + curriculum + loops

Adds decaying intra-layer loops with horizon-1 BPTT and PCGrad over loop
losses. Phase 0 loops the first layer `max_loops` times; phase k loops
`max(1, max_loops - k)` times. First-loop loss is weighted at
`first_loop_weight` to prioritize the deployable single-pass path that gets
used when the layer is no longer the current loop target.

```bash
uv run python main.py train enwik8 --use-iter --use-curriculum --use-loops
```
→ `./data/enwik8-iter-curric-loop4/`

### 5. Iter + curriculum + loops + buckets

Adds phase-based loss bucketing. Each sample is sorted into "new" (unlocked
this phase), "recent" (within `recent_window` phases), or "old" (older).
Three bucket losses are computed; PCGrad runs across them instead of loops;
loops collapse into a weighted sum within each bucket. Counteracts the
exposure imbalance the curriculum creates (easy chunks see ~3× more updates
than hard chunks in a 12-phase run).

```bash
uv run python main.py train enwik8 \
  --use-iter --use-curriculum --use-loops --use-bucket-reweighting
```
→ `./data/enwik8-iter-curric-loop4-buckets/`

### Ablating PCGrad

Replace PCGrad with weighted sum on any multi-task variant:

```bash
uv run python main.py train enwik8 \
  --use-iter --use-curriculum --use-loops --no-use-pcgrad
```

## Sparsity post-training

Iterative magnitude pruning of a trained checkpoint with brief fine-tuning
between sparsity increases. Sweeps the sparsity level until validation
degrades by more than `--target-degradation` (default 0.10 nats); returns the
best-val checkpoint.

```bash
uv run python main.py train enwik8-sparsify \
  --checkpoint data/enwik8-iter-curric-loop4-buckets/best.pt \
  --schedule ascending
```
→ `./data/sparsify/`

Schedule modes (per-layer sparsity distribution):
- `uniform` — same sparsity across layers
- `ascending` — deeper layers more sparse (default; deeper layers tolerate
  more sparsity due to redundancy)
- `descending` — earlier layers more sparse

Prunes the SwiGLU FFN projections `w1`, `w2`, `w3` by default
(`--prune-projections`).

## Resume training

Resume from any saved checkpoint. The checkpoint stores the current phase
and step within phase.

```bash
uv run python main.py train enwik8 \
  --resume data/enwik8-iter-curric/latest.pt \
  --use-iter --use-curriculum
```

To extend the final phase beyond the original budget, bump
`--steps-per-phase`. The cosine schedule reinterprets itself with the new
total, which is usually what you want (LR continues from where it was, not
restarted to peak):

```bash
uv run python main.py train enwik8 \
  --resume data/enwik8-iter-curric/latest.pt \
  --use-iter --use-curriculum \
  --steps-per-phase 14000
```

## Config reference

CLI flags map directly to `TrainConfig` dataclass fields.

### Mode toggles (`--use-X` / `--no-use-X`)

| Flag                    | Default | Requires            |
|-------------------------|---------|---------------------|
| `use_iter`              | False   | —                   |
| `use_curriculum`        | False   | `use_iter`          |
| `use_loops`             | False   | `use_iter`          |
| `use_bucket_reweighting`| False   | `use_curriculum`    |
| `use_pcgrad`            | True    | (only used when ≥2 tasks) |

### Model

`--n-layers`, `--n-heads`, `--d-model`, `--d-head`, `--d-inner`, `--max-seq-len`

### Training

`--steps-per-phase`, `--batch-size`, `--seq-len`, `--lr`, `--weight-decay`,
`--seed`, `--out-dir`, `--resume`

### Looping

`--max-loops`, `--first-loop-weight`, `--lambda-mono`

### Sparsify

`--checkpoint`, `--schedule`, `--sparsity-max`

## Preset configurations

**Small (default, ~14.8M params)** — 14 layers × d_model=256 × d_inner=1024,
seq_len=512. Suitable for Apple Silicon development. Used in the small-scale
experiments and ablations.

**SoTA-comparable (~50M params)** — 12 layers × d_model=512 × d_inner=2048,
seq_len=1024. Suitable for A100 / single-GPU runs; reproduces the parameter
regime of T12 / Transformer-XL / Longformer at 41-44M.

```bash
uv run python main.py train enwik8 \
  --use-iter --use-curriculum \
  --n-layers 12 --d-model 512 --d-head 64 --d-inner 2048 \
  --seq-len 1024 --max-seq-len 1024 \
  --batch-size 64 --steps-per-phase 4000 --lr 4e-4
```

## Project structure

```
mandorla/
├── data/
│   ├── curriculum.py        # GPT-2 PPL + MiniLM scoring; phase chunks; get_batch
│   └── enwik8.py            # Dataset loading
├── models/
│   └── transformer.py       # Pre-norm transformer with `depth` forward arg
├── modules/                 # attention (flex/sdpa, RoPE), ffn (SwiGLU), tblock
├── train/
│   ├── utils.py             # get_batch, evaluate, generate, save/load, cosine_lr, dashboard
│   ├── pcgrad.py            # PCGrad multi-task gradient projection
│   ├── iter_methods.py      # active params, snapshots, loops, buckets
│   ├── pruning.py           # Magnitude pruning helpers
│   ├── enwik8.py            # Main training (config + orchestration)
│   └── sparsify.py          # Sparsity post-training (config + orchestration)
└── utils/
    └── dashboard.py         # HTML auto-refreshing dashboard
```

## Output

Each run writes to its auto-named directory:

- `latest.pt` — most recent checkpoint (every `--eval-every` steps)
- `best.pt` — checkpoint with lowest validation loss
- `dashboard.html` — auto-refreshing dashboard (train/val/bpb curves, layer-
  change bar chart, last sample)
- `dashboard.json` — sidecar metrics

To view the dashboard locally:

```bash
npx serve data/enwik8-iter-curric-loop4-buckets
```

Then open `http://localhost:3000/dashboard.html`. The page auto-refreshes,
so it works for live monitoring during training and from a phone on the same
LAN.

## Output naming convention

`make_out_dir(cfg)` composes the directory name from active flags:

| Flags                                                           | Directory                                        |
|-----------------------------------------------------------------|--------------------------------------------------|
| (none)                                                          | `./data/enwik8-flat/`                            |
| `use_iter`                                                      | `./data/enwik8-iter/`                            |
| `use_iter`, `use_curriculum`                                    | `./data/enwik8-iter-curric/`                     |
| `use_iter`, `use_curriculum`, `use_loops`                       | `./data/enwik8-iter-curric-loop{N}/`             |
| `use_iter`, `use_curriculum`, `use_loops`, `use_bucket_reweighting` | `./data/enwik8-iter-curric-loop{N}-buckets/` |

Pass `--out-dir` to override.