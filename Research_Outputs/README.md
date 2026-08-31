# Graph Neural Networks for Multivariate Time Series Activity Recognition

**Timos Eleftheriou** — MSc Artificial Intelligence, Maastricht University (DACS)
Defended August 2026 · [timoseleftheriou.com](http://www.timoseleftheriou.com) · [github.com/TimosEle23](https://github.com/TimosEle23)

A controlled benchmark of temporal graph neural networks in which **graph
construction is the controlled variable**, across ~2,700 trained models.

---

## What is in this folder

| File | What it is |
|---|---|
| `Master_thesis_report_Timos_Eleftheriou.pdf` | The full thesis |
| `Master_thesis_presentation_Timos_Eleftheriou.pdf` | Defence presentation, 72 slides incl. appendix |
| `phase1_results_BasicMotions_Epilepsy.csv` | Every Phase-1 run — 1,439 records |
| `phase2_results_DSA.csv` | Every Phase-2 run — 170 records, 8-fold LOSO |
| `phase2_summary_table.csv` | Phase-2 aggregated, the table the thesis reports from |
| `computational_cost_benchmark_L4.csv` | Controlled latency and memory benchmark, NVIDIA L4 |

---

## The question

A temporal graph neural network cannot run until an adjacency matrix is supplied,
and body-worn sensors do not come with one. The graph must be constructed, that
construction is a free choice made once before training, and no prior study had
measured what the choice is worth. Because graph and architecture always varied
together, neither could be credited.

This study holds preprocessing, architectures, optimiser, schedule, metrics and
seeds fixed, and varies **only the adjacency**.

## The findings

| Question | Design | Result |
|---|---|---|
| Does node granularity matter? | same tensor reshaped, 45 nodes vs 5 | **+14.5** points |
| Does edge provenance matter? | same 45 nodes, estimated vs uniform prior | **+11.0** points |
| Does the anatomy itself matter? | random topology, matched nodes and edges | **−0.4** points |

- Varying the **graph** moves accuracy **12.1 to 20.6** points. Varying the
  **architecture** moves it **3.2 to 5.7**. The ranges do not overlap.
- Only **4 of 40** construction strategies beat the strongest sequence baseline
  on Epilepsy.
- A sensor graph supplies an **aggregation**, not domain knowledge. The
  random-topology control is the contribution — a negative result, designed
  deliberately because it was the claim most likely to be wrong.
- On the clinically consequential class (seizure), recall moves from **0.612**
  (LSTM) to **0.965** (A3TGCN with a well-chosen graph), while overall accuracy
  differs by less than a point.

## Data and protocol

| Dataset | Phase | Channels | Instances | Classes | Window | Protocol |
|---|---|---|---|---|---|---|
| BasicMotions | 1 | 6 | 80 | 4 | 100 @ 10 Hz | fixed archive split |
| Epilepsy | 1 | 3 | 275 | 4 | 206 @ 16 Hz | fixed archive split |
| DSA | 2 | 45 | 9,120 | 19 | 125 @ 25 Hz | 8-fold leave-one-subject-out |

Five seeds throughout. Phase-2 comparisons are paired across the eight held-out
subjects, with Wilcoxon signed-rank tests corrected for multiplicity.

## Reading the CSVs

One row per (dataset, model, graph mode, seed). `graph_mode` is the controlled
variable. `accuracy_mean` in Phase 2 is averaged over the eight LOSO folds.

**On the Transformer rows.** The raw archives include `transformer`, but the
Transformer is excluded from every comparison reported in the thesis. The
baseline set is deliberately restricted to models that are structural
counterparts of a component under test (thesis §4.4.5). The rows are retained
here so the record is complete rather than filtered, but reproducing the reported
numbers requires excluding them.

**On computational cost.** The thesis reports cost as parameter count. Wall-clock
times recorded during the sweep are not comparable across runs, because
configurations were deliberately co-scheduled on shared GPUs — the evidence for
that is in the thesis appendix. `computational_cost_benchmark_L4.csv` is the
controlled replacement, measured on one idle device with synthetic inputs, and it
carries its own physical-consistency check.

## Code

Full pipeline — preprocessing, graph construction, models, training, evaluation,
and the analysis that produces every reported table and figure — available on
request and at [github.com/TimosEle23](https://github.com/TimosEle23).
