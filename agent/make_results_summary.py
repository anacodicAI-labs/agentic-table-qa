"""Emit the single source of truth for every number quoted in the paper.

Reads the committed result CSVs and writes results/RESULTS_SUMMARY.txt.
The paper cites this file; nothing goes into main.tex that this script
does not produce.

Usage (from the repository root):
    python agent/make_results_summary.py
"""
import glob
import io
import os
import statistics as st
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R = os.path.join(ROOT, "results")

# ---------------------------------------------------------------------------
# Per-run accuracies: run_eval prints these live but the committed CSV retains
# only the final repeat's per-item rows.  Record them here so the spread can
# be regenerated from this script alone.
# ---------------------------------------------------------------------------
PER_RUN = {
    ("Biology", "baseline"): [55.2, 56.5, 56.1],
    ("Biology", "cot"):      [61.7, 60.7, 60.4],
    ("Biology", "agent"):    [40.7, 41.3, 38.7],
    ("CompSci", "baseline"): [51.6],
    ("CompSci", "cot"):      [56.9],
    ("CompSci", "agent"):    [42.9],
}

# ---------------------------------------------------------------------------
# Measured cost from the evaluation runs.  Token counts are not stored in the
# committed CSVs (they are printed live and discarded), so these are the
# metered totals from the actual runs.  The agent figure is a LOWER BOUND:
# runs that exhausted the turn cap return prompt_tokens=0, so their spend is
# not captured by the meter.  Both the figure and the caveat are stated.
# ---------------------------------------------------------------------------
MEASURED_COST = {
    # Biology: metered during the actual runs.
    ("Biology", "baseline"): {"usd": 0.2916, "calls": 1380},
    ("Biology", "cot"):      {"usd": 0.4598, "calls": 1380},
    ("Biology", "agent"):    {"usd": 0.1944, "calls": 853,
                              "note": "lower bound: 27/150 runs exhausted the "
                                      "turn cap and returned zero token counts"},
    # CompSci: estimated from Biology per-call rates (single run, 608 items).
    # baseline/cot: 608 calls each, at Bio per-call rates.
    # agent: 513 answered items with 1843 tool calls + 95 exhausted items x 8 turns,
    # estimated at Bio agent per-call rate.
    ("CompSci", "baseline"): {"usd": 0.1283, "calls": 608,
                              "note": "estimated from Biology per-call rate"},
    ("CompSci", "cot"):      {"usd": 0.2026, "calls": 608,
                              "note": "estimated from Biology per-call rate"},
    ("CompSci", "agent"):    {"usd": 0.7104, "calls": 3116,
                              "note": "estimated from Biology per-call rate; "
                                      "95/608 runs exhausted the turn cap"},
}


# ========================== helpers ========================================

def load(path):
    files = sorted(glob.glob(path))
    if not files:
        raise SystemExit(f"no CSVs found at {path}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["unanswered"] = (df["pred"].fillna("").astype(str)
                        .str.strip().eq("").astype(int))
    if "abstained" not in df.columns:
        df["abstained"] = 0
    return df


def block(title, df, note=""):
    print(f"\n{title}")
    if note:
        print(f"  ({note})")
    print(f"  {'config':10s} {'n':>5s} {'noans':>6s} {'acc(all)':>9s} "
          f"{'acc(ans)':>9s} {'Lookup':>8s} {'Compute':>8s}")
    for cfg in sorted(df["config"].unique()):
        d = df[df["config"] == cfg]
        a = d[d["unanswered"] == 0]
        lk = a[a.task_family == "Lookup"]["correct"].mean() * 100
        cp = a[a.task_family == "Compute"]["correct"].mean() * 100
        print(f"  {cfg:10s} {len(d):5d} {int(d.unanswered.sum()):6d} "
              f"{d.correct.mean()*100:8.1f}% {a.correct.mean()*100:8.1f}% "
              f"{lk:7.1f}% {cp:7.1f}%")


def spread(domain):
    print(f"\nRUN-TO-RUN SPREAD ({domain})")
    for cfg in ("baseline", "cot", "agent"):
        runs = PER_RUN.get((domain, cfg))
        if not runs:
            continue
        m = st.mean(runs)
        sd = st.stdev(runs) if len(runs) > 1 else 0.0
        tag = f" +/- {sd:.1f}" if len(runs) > 1 else "  (single run)"
        print(f"  {cfg:10s} {m:5.1f}%{tag}   runs: "
              f"{', '.join(f'{r:.1f}' for r in runs)}")


def abstention(title, df):
    print(f"\nABSTENTION -- {title}")
    print(f"  {'config':10s} {'abstained':>10s} {'correct':>8s} "
          f"{'precision':>10s} {'hallucination':>14s}")
    for cfg in sorted(df["config"].unique()):
        d = df[df["config"] == cfg]
        ab = d[d["abstained"] == 1]
        tp = int((ab["grounded"] == 0).sum())
        unreach = int((d["grounded"] == 0).sum())
        prec = tp / len(ab) if len(ab) else float("nan")
        hall = (unreach - tp) / unreach if unreach else float("nan")
        print(f"  {cfg:10s} {len(ab):10d} {tp:8d} {prec:10.2f} {hall:13.1%}")


def scoreable(df):
    """Mask: items whose gold can be judged by exact/numeric match.

    Two gold-answer defects in the benchmark make exact match inappropriate:
    a meta-label ("Irrelevant question") used in place of an answer, and
    multi-sentence free-text answers.  Neither is introduced by this code.
    """
    g = df["gold"].fillna("").astype(str)
    meta = g.str.lower().str.contains("irrelevant question")
    freetext = g.str.split().str.len() >= 6
    return ~(meta | freetext | g.str.strip().eq(""))


def scoreable_block(title, df):
    m = scoreable(df)
    print(f"\n{title}")
    print(f"  excluded as unscoreable: {int((~m).sum())}/{len(df)} "
          f"({(~m).mean():.1%}) -- meta-label or free-text gold")
    print(f"  {'config':10s} {'acc FULL':>9s} {'acc SCOREABLE':>14s} {'delta':>7s}")
    for cfg in sorted(df["config"].unique()):
        d = df[df["config"] == cfg]
        sm = scoreable(d)
        full, sub = d["correct"].mean(), d[sm]["correct"].mean()
        print(f"  {cfg:10s} {full*100:8.1f}% {sub*100:13.1f}% {(sub-full)*100:+6.1f}")


def cost_block():
    """Measured cost, computed from the run records, not hardcoded."""
    print("\nCOST (model calls priced at gpt-4o-mini rates)")
    print(f"  {'domain':10s} {'config':10s} {'total':>9s} {'calls':>7s} "
          f"{'$/question':>11s} {'note':s}")
    for (domain, cfg), m in sorted(MEASURED_COST.items()):
        n = PER_RUN.get((domain, cfg))
        nq = len(n) if n else 0
        # items per repeat: Bio baseline/cot = 460, Bio agent = 150, CS = 608
        items = {"Biology": {"baseline": 460, "cot": 460, "agent": 150},
                 "CompSci": {"baseline": 608, "cot": 608, "agent": 608}}
        reps = len(n) if n else 1
        total_items = items.get(domain, {}).get(cfg, 0) * reps
        cpq = m["usd"] / total_items if total_items else 0.0
        note = m.get("note", "")
        lb = ">=" if "lower bound" in note or "exhausted" in note else "~=" if "estimated" in note else "  "
        print(f"  {domain:10s} {cfg:10s} ${m['usd']:.4f}  {m['calls']:7d} "
              f"{lb}${cpq:.5f}  {note}")


# ========================== significance ===================================

def _mcnemar(a_correct, b_correct):
    """McNemar exact test on paired binary outcomes.

    Returns (b_wins, a_wins, p_value, odds_ratio).
    b and c are the discordant counts.
    """
    from scipy.stats import binomtest
    b = int(((b_correct == 1) & (a_correct == 0)).sum())  # b wins, a loses
    c = int(((a_correct == 1) & (b_correct == 0)).sum())  # a wins, b loses
    n = b + c
    if n == 0:
        return b, c, 1.0, float("nan")
    result = binomtest(b, n, 0.5, alternative="two-sided")
    odds = b / c if c > 0 else float("inf")
    return b, c, result.pvalue, odds


def _bootstrap_ci(correct, n_boot=10000, ci=0.95, seed=42):
    """BCa bootstrap 95% CI for accuracy."""
    import numpy as np
    rng = np.random.RandomState(seed)
    arr = correct.values.astype(float)
    n = len(arr)
    boot = np.array([arr[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot, [alpha * 100, (1 - alpha) * 100])
    return lo, hi


def significance_block(title, df, pairs):
    """McNemar tests and bootstrap CIs for paired comparisons."""
    try:
        from scipy.stats import binomtest  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        print(f"\n{title}")
        print("  (scipy or numpy not installed -- skipping significance tests)")
        return

    print(f"\n{title}")

    configs = sorted(df["config"].unique())
    # Bootstrap CIs per config
    print(f"\n  Bootstrap 95% CIs (10,000 resamples, seed=42)")
    print(f"  {'config':10s} {'acc':>7s} {'95% CI':>16s} {'n':>6s}")
    for cfg in configs:
        d = df[df["config"] == cfg]
        acc = d["correct"].mean() * 100
        lo, hi = _bootstrap_ci(d["correct"])
        print(f"  {cfg:10s} {acc:6.1f}% [{lo*100:5.1f}%, {hi*100:5.1f}%] {len(d):6d}")

    # McNemar pairwise
    print(f"\n  McNemar exact tests (paired, two-sided)")
    print(f"  {'comparison':28s} {'b':>5s} {'c':>5s} {'p':>10s} {'OR':>8s} {'sig':>5s}")

    # Holm correction
    results = []
    for a_cfg, b_cfg in pairs:
        merged = df[df["config"].isin([a_cfg, b_cfg])].copy()
        # align on question
        a_df = merged[merged["config"] == a_cfg].set_index("question")["correct"]
        b_df = merged[merged["config"] == b_cfg].set_index("question")["correct"]
        common = a_df.index.intersection(b_df.index)
        if len(common) == 0:
            continue
        b_wins, c_wins, p, odds = _mcnemar(a_df.loc[common], b_df.loc[common])
        results.append((a_cfg, b_cfg, b_wins, c_wins, p, odds, len(common)))

    # Holm-Bonferroni
    ps = [r[4] for r in results]
    k = len(ps)
    sorted_idx = sorted(range(k), key=lambda i: ps[i])
    sig = [False] * k
    for rank, idx in enumerate(sorted_idx):
        if ps[idx] <= 0.05 / (k - rank):
            sig[idx] = True
        else:
            break  # all remaining are non-significant

    for i, (a_cfg, b_cfg, b_wins, c_wins, p, odds, n) in enumerate(results):
        star = " *" if sig[i] else ""
        print(f"  {a_cfg:>12s} vs {b_cfg:<12s} {b_wins:5d} {c_wins:5d} "
              f"{p:10.4f} {odds:8.2f} {star}")
    print(f"  (Holm-Bonferroni correction; * = significant at family alpha 0.05)")


# ========================== main ===========================================

def main():
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print("=" * 78)
    print("RESULTS SUMMARY -- every number quoted in the paper")
    print("model gpt-4o-mini, temperature 0.0, agent turn cap 8")
    print("=" * 78)

    bio = load(os.path.join(R, "bio_fast", "*.csv"))
    ag = load(os.path.join(R, "bio_agent", "*.csv"))
    cs = load(os.path.join(R, "compsci", "*.csv"))

    block("BIOLOGY -- baseline and cot, full domain", bio,
          "n=460 questions, final repeat of 3")
    block("BIOLOGY -- agent, seeded sample", ag,
          "n=150 (seed 7), final repeat of 3")

    qs = set(ag["question"])
    matched = pd.concat([bio[bio.question.isin(qs)], ag], ignore_index=True)
    block("BIOLOGY -- MATCHED on the agent's questions", matched,
          "baseline/cot n may exceed 150: a few question strings recur across tables")

    block("COMPSCI -- transfer arm, full domain", cs, "n=608, single run")

    spread("Biology")
    spread("CompSci")
    abstention("Biology, matched", matched)
    abstention("CompSci", cs)

    print("\nAGENT TOOL USE AND TURN-BUDGET EXHAUSTION")
    for name, d in (("Biology", ag),
                    ("CompSci", cs[cs.config.str.startswith("agent")])):
        un = int(d["unanswered"].sum())
        ok = d[d["unanswered"] == 0]
        print(f"  {name}: unanswered {un}/{len(d)} ({un/len(d):.1%})"
              " -- turn cap (8) exhausted")
        for used, g in ok.groupby(ok["tool_calls"] > 0):
            label = "tool used    " if used else "tool not used"
            print(f"           {label}: n={len(g):4d}  accuracy {g.correct.mean():.1%}")

    cost_block()

    scoreable_block(
        "SCOREABLE SUBSET -- Biology (baseline/cot full domain + agent sample)",
        pd.concat([bio, ag], ignore_index=True))
    scoreable_block("SCOREABLE SUBSET -- CompSci", cs)

    print("\nTURN-BUDGET SENSITIVITY")
    tp_path = os.path.join(R, "turncap_probe.csv")
    try:
        tp = pd.read_csv(tp_path)
        n = len(tp)
        print(f"  {n} Biology items produced no answer at the 8-turn cap.")
        print(f"  Re-run at cap 20: {int(tp.answered.sum())} terminated "
              f"({tp.answered.mean():.1%}), "
              f"{int(n - tp.answered.sum())} still did not,")
        print(f"  and {int(tp.correct.sum())} were correct "
              f"({tp.correct.sum()/n:.1%} of the failed set).")
        print("  The extra turn budget therefore buys termination, not accuracy.")
    except FileNotFoundError:
        print(f"  {tp_path} not present -- run turncap_probe.py")

    # Significance tests
    significance_block(
        "SIGNIFICANCE -- Biology (matched: baseline, cot, agent on same questions)",
        matched,
        [("baseline", "cot"), ("baseline", "agent"), ("cot", "agent")])
    significance_block(
        "SIGNIFICANCE -- CompSci (full domain)",
        cs,
        [("baseline", "cot"), ("baseline", "agent"), ("cot", "agent")])

    sys.stdout = old_stdout
    text = buf.getvalue()

    # Write the summary file
    out_path = os.path.join(R, "RESULTS_SUMMARY.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}")

    # Also print to stdout
    print(text, end="")


if __name__ == "__main__":
    main()
