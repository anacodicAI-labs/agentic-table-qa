# agentic-table-qa

Code and evaluation harness for **"Do Agentic Components Earn Their Cost? Reasoning,
Tools and Abstention in Table Question Answering."**

A tool-using agent answers questions whose answers live in a scientific table. The
agent decides for itself whether to invoke a tool or answer directly, and abstains
when it judges the table does not contain the answer. The repository exists so that
every number in the paper can be reproduced from a clean clone, offline.

## What is measured, and why

The answer to each question is a single cell or a single number, so the metric is
exact match with numeric awareness — not BLEU, and not a retrieval metric, because
nothing is retrieved: the whole table is supplied with the question.

Questions fall into two families, and results are **never** reported as a blend:

| family | the answer is | example |
|---|---|---|
| **Lookup** | a cell in the table | *"forward primer sequence for Exon 3?"* |
| **Compute** | derived from cells | *"what percentage of oocytes had Simple Errors?"* |

Averaging a largely solved family with an unsolved one hides both.

## Layout

```
agent/   the system
  scitableqa_loader.py   loads (question, gold, table) triples from a local cache
  table_tools.py         CSV -> markdown / DataFrame
  scorer.py              numeric-aware exact match
  agent.py               plain model call (baseline, cot)
  agent_sdk.py           the tool-using agent (OpenAI Agents SDK)
  run_eval.py            runs the configurations and writes result CSVs
  01..05_*.ipynb         walkthrough notebooks: data, scoring, baseline, agent, experiment

eval/    the analysis
  analyze_results.py     accuracy by family, ablation delta, abstention calibration
  01..03_*.ipynb         run the experiment, analyse results, error analysis
  START_HERE.md          setup and tasks
```

## Setup

```bash
pip install openai openai-agents pandas matplotlib nbformat nest_asyncio
echo "OPENAI_API_KEY=sk-..." > .env          # or GROQ_API_KEY
```

The benchmark (SciTableQA, MIT licence) is cached locally on first use, after which
evaluation makes no network calls beyond the model API.

## Reproducing the paper's numbers

```bash
cd agent

# main arm: biomedical case study
BUDGET_USD=3.00 python run_eval.py --domain Biology \
    --configs baseline,cot,agent --repeats 3 --out-dir results/biology

# transfer arm: a second domain
BUDGET_USD=1.50 python run_eval.py --domain CompSci \
    --configs baseline,cot,agent --repeats 1 --out-dir results/compsci
```

Every run is executed under an enforced spend ceiling (`BUDGET_USD`); the run stops
and writes partial results rather than exceeding it.

Then:

```bash
cd ../eval && python analyze_results.py
```

## The three reporting rules

1. **Never blend Lookup and Compute.** Report the split, always.
2. **Never claim a difference smaller than the run-to-run spread.** Hence `--repeats 3`.
3. **Report cost beside accuracy.** A gain is stated per dollar, not in isolation.

## Abstention

The agent may reply `INSUFFICIENT`. That decision is scored against whether the gold
answer is actually reachable from the table (the `grounded` column):

```
                    gold IS in table      gold is NOT in table
 agent abstained    false alarm           correct abstention
 agent answered     scored normally       hallucination
```

A run restricted to grounded items cannot measure this — every abstention is a false
alarm by construction. Run without `--grounded-only` to obtain the calibration figures.

## Benchmark note

Not every gold answer is reachable from the table as extracted, which caps attainable
accuracy. Measured grounded rates:

| domain | Lookup | Compute |
|---|---|---|
| Biology | 80% | 50% |
| CompSci | 84% | 26% |

Results are reported on both the verifiable subset and the full set so the ceiling is
explicit.

## Development history

Developed in [`anacodic-agentic-cookbook`](https://github.com/anacodicAI-labs/anacodic-agentic-cookbook)
under `02-experiments/`. This repository is the citable artifact for the paper.

## Licence

Code: MIT. SciTableQA is distributed by its authors under the MIT licence.
