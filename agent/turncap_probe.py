"""Does raising the turn cap convert non-termination into answers?

The paper reports a non-termination rate at a stated turn cap. That claim is only
meaningful if the cap can be moved and the effect measured, so this probe re-runs
exactly the items that produced no answer at cap 8 with the cap raised, and reports
how many then terminate and how many of those are correct.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scitableqa_loader as L          # noqa: E402
import agent_sdk                       # noqa: E402
from scorer import is_correct          # noqa: E402

CAP = int(os.environ.get("MAX_TURNS", "20"))
prev = pd.read_csv("results_v2/bio_agent/agent.csv")
prev["un"] = prev.pred.fillna("").astype(str).str.strip().eq("")
failed = prev[prev.un]
trips = {t.question: t for t in L.load_domain("Biology")}

print(f"re-running {len(failed)} items that produced no answer at cap 8, now at cap {CAP}")
rows = []
for i, (_, r) in enumerate(failed.iterrows(), 1):
    t = trips.get(r.question)
    if t is None:
        continue
    out = agent_sdk.run_agent(t.question, t.table_csv, False, 0.0)
    final = str(out.get("final") or "").strip()
    rows.append({"question": t.question, "gold": t.gold, "pred": final,
                 "answered": int(bool(final)),
                 "correct": int(bool(final)) and int(is_correct(final, t.gold, t.question)),
                 "task_family": t.task_family, "error": out.get("error", "")})
    if i % 5 == 0:
        print(f"  {i}/{len(failed)}", flush=True)

d = pd.DataFrame(rows)
d.to_csv("results_v2/turncap_probe.csv", index=False)
n = len(d)
print(f"\nat cap {CAP}, of {n} previously non-terminating items:")
print(f"  now answered      {int(d.answered.sum()):3d}  ({d.answered.mean():.1%})")
print(f"  still no answer   {int(n - d.answered.sum()):3d}")
print(f"  answered AND correct {int(d.correct.sum()):3d}  "
      f"({d.correct.sum()/n:.1%} of the failed set)")
