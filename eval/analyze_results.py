"""Analyze table-QA result CSVs: accuracy by reasoning type + ablation delta + a figure.

By default reads the committed results at results/ (relative to the repository root).
Override with --input-dir to point at a different directory.

Prints per-config accuracy and, if both a 'baseline' and an 'agent' config are present,
the delta.  Reports the Lookup / Compute split from task_family, and abstention
calibration when the relevant columns are present.  Saves a bar chart to
results/accuracy_by_type.png.  No model calls -- pure analysis of outputs.
"""
import argparse
import glob
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_IN = os.path.join(ROOT, "results")


def load(input_dir: str) -> pd.DataFrame:
    # Recurse into subdirectories to find all CSVs
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    # Exclude summary files
    files = [f for f in files if "turncap_probe" not in f
             and "RESULTS_SUMMARY" not in f]
    if not files:
        raise SystemExit(
            f"no result CSVs found under {input_dir}.\n"
            f"Run the experiment first (see agent/run_eval.py) or point\n"
            f"--input-dir at a directory containing result CSVs."
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_IN,
                    help="directory containing result CSVs (default: results/)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the figure (default: same as input-dir)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.input_dir
    os.makedirs(out_dir, exist_ok=True)

    df = load(args.input_dir)
    configs = sorted(df["config"].unique())
    print(f"loaded {len(df)} rows across configs: {configs}\n")

    # Use task_family for the split if available, else fall back to reasoning_type
    group_col = "task_family" if "task_family" in df.columns else "reasoning_type"

    for cfg in configs:
        d = df[df["config"] == cfg]
        acc = d["correct"].mean()
        print(f"[{cfg}] overall accuracy = {d['correct'].sum()}/{len(d)} = {acc:.1%}")
        by = d.groupby(group_col)["correct"].agg(["sum", "count"])
        for rt, row in by.iterrows():
            print(f"    {str(rt):28s} {int(row['sum'])}/{int(row['count'])}")
        print()

    # Dual accuracy: if both full-set and scoreable can be shown
    if "gold" in df.columns:
        g = df["gold"].fillna("").astype(str)
        meta = g.str.lower().str.contains("irrelevant question")
        freetext = g.str.split().str.len() >= 6
        scoreable_mask = ~(meta | freetext | g.str.strip().eq(""))
        excluded = int((~scoreable_mask).sum())
        if excluded > 0:
            print(f"=== SCOREABLE SUBSET (excluded {excluded}/{len(df)} "
                  f"meta-label / free-text golds) ===")
            for cfg in configs:
                d = df[df["config"] == cfg]
                sm = scoreable_mask[d.index]
                full = d["correct"].mean()
                sub = d.loc[sm, "correct"].mean() if sm.any() else float("nan")
                print(f"  [{cfg}] full {full:.1%}  scoreable {sub:.1%}  "
                      f"delta {(sub - full)*100:+.1f}pp")
            print()

    # Abstention calibration
    if {"abstained", "grounded"} <= set(df.columns):
        print("=== ABSTENTION CALIBRATION ===")
        print(f"  {'config':12s} {'abstained':>10s} {'correct':>8s} "
              f"{'false alarm':>12s} {'precision':>10s}")
        for cfg in configs:
            d = df[df["config"] == cfg]
            ab = d[d["abstained"] == 1]
            if len(ab) == 0:
                print(f"  {cfg:12s} {'0':>10s}")
                continue
            tp = int((ab["grounded"] == 0).sum())
            fp = len(ab) - tp
            prec = tp / len(ab) if len(ab) else float("nan")
            print(f"  {cfg:12s} {len(ab):10d} {tp:8d} {fp:12d} {prec:10.2f}")
        if (df["grounded"] == 1).all():
            print("  NOTE: every item is grounded -- a correct abstention is impossible"
                  " here. Re-run without --grounded-only to measure calibration.")
        print()

    # Figure: accuracy by task family (or reasoning type), one bar group per config
    piv = df.pivot_table(index=group_col, columns="config",
                         values="correct", aggfunc="mean")
    ax = piv.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Accuracy by {group_col.replace('_', ' ')}")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "accuracy_by_type.png")
    plt.savefig(fig_path, dpi=120)
    print(f"saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
