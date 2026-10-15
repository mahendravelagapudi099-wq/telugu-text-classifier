"""Run the Phase 2 experiment grid sequentially: models x seeds.

Writes one JSON per run to results/runs/. Long-running by design; launch
detached (nohup) on a GPU machine and poll results/runs/ for progress.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config  # noqa: E402


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def run_one(model: str, seed: int, lr: float | None, output_root: Path, runs_dir: Path) -> Path:
    cfg = load_config()["training"]
    name = f"{short_name(model)}_seed{seed}" + (f"_lr{lr:g}" if lr else "")
    out_dir = output_root / name
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_transformer.py"),
        "--model", model,
        "--seed", str(seed),
        "--output-dir", str(out_dir),
        "--lr", str(lr) if lr else str(cfg["learning_rate"]),
    ]
    print(f">>> {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise SystemExit(f"training failed: {name}")
    summary_path = out_dir / "run_summary.json"
    runs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(summary_path, runs_dir / f"{name}.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        f"[{name}] done in {time.time() - start:.0f}s, "
        f"best val macro-F1 {summary['best_val_macro_f1']:.4f}",
        flush=True,
    )
    return summary_path


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=cfg["training"]["models"])
    parser.add_argument("--seeds", nargs="*", type=int, default=cfg["training"]["seeds"])
    parser.add_argument("--lr", type=float, default=None, help="override LR for all runs")
    parser.add_argument("--output-root", type=Path, default=Path(cfg["paths"]["outputs"]))
    args = parser.parse_args()

    runs_dir = Path(cfg["paths"]["results"]) / "runs"
    for model in args.models:
        for seed in args.seeds:
            run_one(model, seed, args.lr, args.output_root, runs_dir)
    print("ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
