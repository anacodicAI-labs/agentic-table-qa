"""Analyze table-QA result CSVs: accuracy by reasoning type + ablation delta + a figure.

Reads every *.csv in results-input/ (columns: question, gold, pred, correct,
reasoning_type, model, config). Prints per-config accuracy and, if both a
'naive' and a 'table_aware' config are present, the ablation delta. Saves a bar
chart to results/accuracy_by_type.png. No model calls — pure analysis of outputs.
"""
import glob
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
IN = os.path.join(HERE, "results-input")
OUT = os.path.join(HERE, "results")


def load() -> pd.DataFrame:
    files = glob.glob(os.path.join(IN, "*.csv"))
    if not files:
        raise SystemExit(
            f"no CSVs found in {IN}. Put a result file there first "
            f"(a sample file is provided — see START_HERE.md)."
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    configs = sorted(df["config"].unique())
    print(f"loaded {len(df)} rows across configs: {configs}\n")

    # An agent run can end without producing an answer (e.g. the turn cap is
    # exhausted). That is an agent failure, not a wrong answer. Report accuracy both
    # including and excluding those items -- never one alone. Older CSVs predate the
    # `unanswered` column, so derive it from an empty prediction.
    if "unanswered" not in df.columns:
        df["unanswered"] = (df["pred"].fillna("").astype(str).str.strip() == "").astype(int)
    print(f"{'config':12s} {'no answer':>10s} {'acc (all)':>10s} {'acc (answered)':>15s}")
    for cfg in configs:
        d = df[df["config"] == cfg]
        answered = d[d["unanswered"] == 0]
        exc = answered["correct"].mean() if len(answered) else float("nan")
        print(f"{cfg:12s} {int(d['unanswered'].sum()):10d} {d['correct'].mean():9.1%} "
              f"{exc:14.1%}")
        if "error" in d.columns:
            reasons = (d.loc[d["unanswered"] == 1, "error"].fillna("unknown")
                       .astype(str).str.split(":").str[0].value_counts().head(3))
            for reason, k in reasons.items():
                print(f"{'':12s} {k:10d}  {reason[:48]}")
    print()

    for cfg in configs:
        d = df[df["config"] == cfg]
        acc = d["correct"].mean()
        print(f"[{cfg}] overall accuracy = {d['correct'].sum()}/{len(d)} = {acc:.1%}")
        by = d.groupby("reasoning_type")["correct"].agg(["sum", "count"])
        for rt, row in by.iterrows():
            print(f"    {rt:28s} {int(row['sum'])}/{int(row['count'])}")
        print()

    # ablation delta if both configs present
    if "naive" in configs and "table_aware" in configs:
        print("=== ABLATION: naive -> table_aware (accuracy by reasoning type) ===")
        piv = df[df.config.isin(["naive", "table_aware"])].pivot_table(
            index="reasoning_type", columns="config", values="correct", aggfunc="mean")
        print((piv * 100).round(1))

    # abstention calibration: the agent decides on its own that it cannot answer.
    # Scoring that decision needs the `grounded` flag, i.e. whether the gold answer
    # is actually reachable from the table. On a grounded-only run every abstention
    # is a false alarm by construction, so the rates below are only meaningful when
    # ungrounded items are present.
    if {"abstained", "grounded"} <= set(df.columns):
        print("=== ABSTENTION: is the agent's refusal to answer well calibrated? ===")
        print(f"  {'config':12s} {'abstained':>10s} {'correct':>8s} {'false alarm':>12s}"
              f" {'precision':>10s} {'recall':>7s} {'hallucination':>14s}")
        for cfg in configs:
            d = df[df["config"] == cfg]
            ab = d[d["abstained"] == 1]
            tp = int((ab["grounded"] == 0).sum())       # abstained, gold unreachable
            fp = len(ab) - tp                           # abstained, gold was there
            unreach = int((d["grounded"] == 0).sum())
            hall = unreach - tp                         # answered anyway
            prec = tp / len(ab) if len(ab) else float("nan")
            rec = tp / unreach if unreach else float("nan")
            hrate = hall / unreach if unreach else float("nan")
            print(f"  {cfg:12s} {len(ab):10d} {tp:8d} {fp:12d} {prec:10.2f} "
                  f"{rec:7.2f} {hrate:13.1%}")
        if (df["grounded"] == 1).all():
            print("  NOTE: every item is grounded - a correct abstention is impossible"
                  " here. Re-run without --grounded-only to measure calibration.")
        print()

    # figure: accuracy by reasoning type, one bar group per config
    piv = df.pivot_table(index="reasoning_type", columns="config",
                         values="correct", aggfunc="mean")
    ax = piv.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Accuracy by reasoning type")
    plt.tight_layout()
    fig_path = os.path.join(OUT, "accuracy_by_type.png")
    plt.savefig(fig_path, dpi=120)
    print(f"\nsaved figure -> {fig_path}")


if __name__ == "__main__":
    main()
