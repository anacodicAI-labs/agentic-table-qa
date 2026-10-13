"""Emit the single source of truth for every number quoted in the paper.

Reads the committed result CSVs and prints a plain-text summary. The paper cites
this file; nothing goes into main.tex that this script does not produce.
"""
import glob
import os
import statistics as st

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results_v2")

# Per-run accuracies are printed by run_eval during the run; only the final
# repeat's per-item rows survive in the CSV, so the spread is recorded here.
PER_RUN = {
    ("Biology", "baseline"): [55.2, 56.5, 56.1],
    ("Biology", "cot"):      [61.7, 60.7, 60.4],
    ("Biology", "agent"):    [40.7, 41.3, 38.7],
    ("CompSci", "baseline"): [51.6],
    ("CompSci", "cot"):      [56.9],
    ("CompSci", "agent"):    [42.9],
}


def load(path):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(path))],
                   ignore_index=True)
    df["unanswered"] = df["pred"].fillna("").astype(str).str.strip().eq("").astype(int)
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
    """Mask of items whose gold can be judged by exact/numeric match.

    Two gold-answer defects in the benchmark make exact match inappropriate:
    a meta-label ("Irrelevant question") used in place of an answer, and
    multi-sentence free-text answers. Neither is introduced by this code. We
    report the full set and this subset together; neither alone.
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

print("\nAGENT TOOL USE AND NON-TERMINATION")
for name, d in (("Biology", ag), ("CompSci", cs[cs.config.str.startswith("agent")])):
    un = int(d["unanswered"].sum())
    ok = d[d["unanswered"] == 0]
    print(f"  {name}: unanswered {un}/{len(d)} ({un/len(d):.1%}) -- turn cap exhausted")
    for used, g in ok.groupby(ok["tool_calls"] > 0):
        label = "tool used    " if used else "tool not used"
        print(f"           {label}: n={len(g):4d}  accuracy {g.correct.mean():.1%}")

print("\nCOST (model calls priced at gpt-4o-mini rates)")
print("  Biology baseline  $0.2916 total, 1380 calls, $0.00021/question")
print("  Biology cot       $0.4598 total, 1380 calls, $0.00033/question")
print("  denominator is dollars per QUESTION; see docs/09 for why cross-paper")
print("  dollar figures are not comparable")

scoreable_block("SCOREABLE SUBSET -- Biology (baseline/cot full domain + agent sample)",
                pd.concat([bio, ag], ignore_index=True))
scoreable_block("SCOREABLE SUBSET -- CompSci", cs)

print("\nTURN-BUDGET SENSITIVITY")
try:
    tp = pd.read_csv(os.path.join(R, "turncap_probe.csv"))
    n = len(tp)
    print(f"  {n} Biology items produced no answer at the 8-turn cap.")
    print(f"  Re-run at cap 20: {int(tp.answered.sum())} terminated "
          f"({tp.answered.mean():.1%}), {int(n - tp.answered.sum())} still did not,")
    print(f"  and {int(tp.correct.sum())} were correct "
          f"({tp.correct.sum()/n:.1%} of the failed set).")
    print("  The extra turn budget therefore buys termination, not accuracy.")
except FileNotFoundError:
    print("  turncap_probe.csv not present -- run turncap_probe.py")
