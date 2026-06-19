import json
from datetime import datetime
from pathlib import Path
from typing import Any


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta http-equiv="refresh" content="10">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; max-width: 1100px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2em; }}
  h2 {{ margin-top: 2em; font-size: 1.1em; }}
  h3 {{ margin: 0 0 0.5em; font-size: 0.95em; font-weight: 500; color: #555; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 2em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }}
  canvas {{ max-height: 280px; }}
  pre {{ background: #f5f5f5; padding: 1em; border-radius: 6px; white-space: pre-wrap; font-size: 0.85em; max-height: 400px; overflow-y: auto; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">step {step:,} | updated {timestamp}</p>
<div class="grid">
  <div><h3>train loss</h3><canvas id="train_loss"></canvas></div>
  <div><h3>val loss (95% CI)</h3><canvas id="val_loss"></canvas></div>
  <div><h3>bpb (95% CI)</h3><canvas id="bpb"></canvas></div>
  <div><h3>learning rate</h3><canvas id="lr"></canvas></div>
  <div><h3>throughput</h3><canvas id="tps"></canvas></div>
  <div><h3>weight Δ since phase start</h3><canvas id="layer_changes"></canvas></div>
</div>
<h2>latest sample</h2>
<pre>{sample}</pre>
<script>
const series = {data};
const xy = (xs, ys) => xs.map((x, i) => ({{x, y: ys[i]}}));

function lineChart(id, datasets, yLabel) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true, animation: false,
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'step' }} }},
        y: {{ title: {{ display: true, text: yLabel }} }}
      }},
      plugins: {{ legend: {{ labels: {{ filter: i => !i.text.startsWith('_') }} }} }},
      elements: {{ point: {{ radius: 0 }} }}
    }}
  }});
}}

function ciBand(steps, mean, lo, hi, color, fillColor, label) {{
  return [
    {{ label: '_lo', data: xy(steps, lo), borderColor: 'transparent', backgroundColor: fillColor, fill: '+1', pointRadius: 0 }},
    {{ label: '_hi', data: xy(steps, hi), borderColor: 'transparent', pointRadius: 0 }},
    {{ label, data: xy(steps, mean), borderColor: color, borderWidth: 1.5 }},
  ];
}}

lineChart('train_loss', [
  {{ label: 'train loss', data: xy(series.train_steps, series.train_loss), borderColor: '#3b82f6', borderWidth: 1.5 }}
], 'loss');

lineChart('val_loss', ciBand(
  series.eval_steps, series.val_loss, series.val_lo, series.val_hi,
  '#ef4444', 'rgba(239, 68, 68, 0.15)', 'val loss'
), 'loss');

lineChart('bpb', ciBand(
  series.eval_steps, series.val_bpb, series.bpb_lo, series.bpb_hi,
  '#10b981', 'rgba(16, 185, 129, 0.15)', 'bpb'
), 'bpb');

lineChart('lr', [
  {{ label: 'lr', data: xy(series.train_steps, series.lr), borderColor: '#8b5cf6', borderWidth: 1.5 }}
], 'lr');

lineChart('tps', [
  {{ label: 'tok/s', data: xy(series.train_steps, series.tps), borderColor: '#f59e0b', borderWidth: 1.5 }}
], 'tok/s');

const lcHistory = series.layer_changes || [];
const latestLc = lcHistory.length > 0 ? lcHistory[lcHistory.length - 1] : {{}};
const lcKeys = Object.keys(latestLc).sort((a, b) => +a - +b);
new Chart(document.getElementById('layer_changes'), {{
  type: 'bar',
  data: {{
    labels: lcKeys.map(k => 'L' + k),
    datasets: [{{
      label: 'relative Δ',
      data: lcKeys.map(k => latestLc[k]),
      backgroundColor: '#8b5cf6',
    }}],
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ beginAtZero: true, title: {{ display: true, text: '||Δw|| / ||w₀||' }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""


def _escape(s: str) -> str:
  return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_atomic(path: Path, content: str) -> None:
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(content, encoding="utf-8")
  tmp.replace(path)


class Dashboard:
  def __init__(self, out_dir: str | Path, title: str = "Training") -> None:
    self.dir = Path(out_dir)
    self.html_path = self.dir / "dashboard.html"
    self.json_path = self.dir / "dashboard.json"
    self.title = title
    self.step = 0
    self.last_sample = ""
    self.data: dict[str, list[Any]] = {
      "train_steps": [], "train_loss": [], "lr": [], "tps": [],
      "eval_steps": [],
      "val_loss": [], "val_lo": [], "val_hi": [],
      "val_bpb": [], "bpb_lo": [], "bpb_hi": [],
      "layer_change_steps": [], "layer_changes": [],
    }

  def load(self) -> None:
    if not self.json_path.exists():
      return
    state = json.loads(self.json_path.read_text())
    self.data = state["data"]
    self.step = state["step"]
    self.last_sample = state["last_sample"]

  def log_train(self, step: int, loss: float, lr: float, tps: float) -> None:
    self.step = step
    self.data["train_steps"].append(step)
    self.data["train_loss"].append(loss)
    self.data["lr"].append(lr)
    self.data["tps"].append(tps)

  def log_eval(self, step: int, val_loss: float, val_lo: float, val_hi: float,
               bpb: float, bpb_lo: float, bpb_hi: float, sample: str = "") -> None:
    self.step = step
    self.data["eval_steps"].append(step)
    self.data["val_loss"].append(val_loss)
    self.data["val_lo"].append(val_lo)
    self.data["val_hi"].append(val_hi)
    self.data["val_bpb"].append(bpb)
    self.data["bpb_lo"].append(bpb_lo)
    self.data["bpb_hi"].append(bpb_hi)
    if sample:
      self.last_sample = sample

  def log_layer_changes(self, step: int, changes: dict[int, float]) -> None:
    self.step = step
    self.data["layer_change_steps"].append(step)
    # JSON keys must be strings
    self.data["layer_changes"].append({str(k): v for k, v in changes.items()})

  def flush(self) -> None:
    self.dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(self.json_path, json.dumps({
      "data": self.data, "step": self.step, "last_sample": self.last_sample,
    }))
    _write_atomic(self.html_path, _TEMPLATE.format(
      title=self.title,
      step=self.step,
      timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      sample=_escape(self.last_sample) or "(no sample yet)",
      data=json.dumps(self.data),
    ))