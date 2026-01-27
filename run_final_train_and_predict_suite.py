#!/usr/bin/env python3
"""Run final training + unknown-gene prediction for multiple feature sets.

The reviewer remediation work often requires producing multiple ranked candidate
lists (e.g., baseline vs ablations vs +conservation features). This helper wraps
`ml/train_final_and_predict_unknowns.py` and runs it once per feature set,
placing outputs into subdirectories.

Example
-------
python ml/run_final_train_and_predict_suite.py \
  --data ml/data/prepared_for_modeling_augmented.csv \
  --out ml/outputs/final_predictions_suite \
  --feature_sets principled_expression_no_position_no_conservation \
               principled_expression_no_position_plus_conservation \
  --models logreg rf xgb --make_ensemble --save_all_predictions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature_sets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["rf"], choices=["logreg", "rf", "xgb"])

    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--make_ensemble", action="store_true")
    ap.add_argument("--save_all_predictions", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "train_final_and_predict_unknowns.py"
    if not script.exists():
        raise SystemExit(f"Could not locate {script}")

    for fs in a.feature_sets:
        fs_out = out_root / fs
        fs_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script),
            "--data",
            a.data,
            "--out",
            str(fs_out),
            "--feature_set",
            fs,
            "--models",
            *a.models,
            "--cv_folds",
            str(a.cv_folds),
            "--n_iter",
            str(a.n_iter),
            "--random_state",
            str(a.random_state),
            "--n_jobs",
            str(a.n_jobs),
        ]
        if a.make_ensemble:
            cmd.append("--make_ensemble")
        if a.save_all_predictions:
            cmd.append("--save_all_predictions")

        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("✓ Final training + prediction suite complete")
    print("Outputs written to:", out_root.resolve())


if __name__ == "__main__":
    main()
