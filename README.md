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

These are the exact commands behind the committed CSVs in `results/`. All runs use
`gpt-4o-mini` at temperature 0.0, with the agent's turn cap at 8.

```bash
cd agent

# Biology, baseline and cot, full domain, three repeats
BUDGET_USD=1.20 python run_eval.py --domain Biology \
    --configs baseline,cot --repeats 3 --limit 460 \
    --out-dir results_v2/bio_fast

# Biology, agent, seeded 150-question sample, three repeats
BUDGET_USD=1.50 python run_eval.py --domain Biology \
    --configs agent --repeats 3 --limit 150 --sample-seed 7 \
    --out-dir results_v2/bio_agent

# CompSci transfer arm, full domain, single run
BUDGET_USD=1.50 python run_eval.py --domain CompSci \
    --configs baseline,cot,agent --repeats 1 --limit 608 \
    --out-dir results_v2/compsci
```

The agent config is run separately because it is multi-turn and far slower than the
plain calls; keeping it in its own process stops it blocking the other repeats.

Then regenerate the summary that the paper cites:

```bash
python make_results_summary.py        # -> results/RESULTS_SUMMARY.txt
cd ../eval && python analyze_results.py
```

`results/RESULTS_SUMMARY.txt` is the single source of truth: no number appears in the
paper that this script does not produce.

## Results

Accuracy on answered items, by task family. Biology figures are the mean of three
runs; CompSci is a single run.

| Domain | Config | Overall | Lookup | Compute | No answer |
|---|---|---|---|---|---|
| Biology | baseline | 55.9 ± 0.7 | 77.6 | 40.2 | 0 |
| Biology | cot | **60.9 ± 0.7** | 76.5 | **48.5** | 0 |
| Biology | agent | 40.2 ± 1.4 | 58.2 | 38.2 | 27 / 150 (18.0%) |
| CompSci | baseline | 51.6 | 66.9 | 40.2 | 0 |
| CompSci | cot | **56.9** | 66.9 | **49.4** | 0 |
| CompSci | agent | 42.9 | 59.0 | 44.3 | 95 / 608 (15.6%) |

Three findings, both domains:

1. **Step-by-step prompting pays, and only on Compute.** +5.0 points on Biology
   against a 0.7-point spread, with the entire gain on derived-answer questions and
   none on lookup.
2. **Tools and autonomy do not.** The agent is last in both domains, losing most
   heavily on Lookup — it writes code to retrieve values a plain call simply reads.
   Among answered items it invoked its tool on 123/123 (Biology) and 512/513
   (CompSci), so it is not exercising discretion.
3. **Non-termination is the real failure mode.** 15–18% of agent runs end with
   `Max turns (8) exceeded` and no answer. Scoring those as wrong answers would
   understate the agent by a further 7.6–8.5 points and misattribute a control-flow
   failure to reasoning.

Abstention precision falls monotonically as machinery is added (Biology
0.54 → 0.29 → 0.21; CompSci 0.77 → 0.72 → 0.50): the plain call is the most reliable
at knowing when not to answer.

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
