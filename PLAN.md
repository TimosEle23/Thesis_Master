# MASTER EXECUTION PLAN
## Benchmarking TGNNs for Multivariate Time Series Classification in Wearable HAR

> Generated: 2026-03-31 | Status: Phase 1 complete, Phase 2 in-progress

---

## 0. CURRENT STATE AUDIT

### What's Done
| Item | Status |
|------|--------|
| Phase 1 experiments (BasicMotions + Epilepsy) | **120/120 complete** (6 models × 2 graph modes × 5 seeds × 2 datasets) |
| Phase 1 analysis (tables, Wilcoxon, confusion matrices) | **Done** |
| Phase 2 DSA LSTM baseline (seed 42, 8 folds) | **Done** |
| Phase 2 DSA LSTM baseline (seed 123, 6 folds) | **Partial** (2 folds missing) |
| Code: all models, data loaders, graphs, training | **Production-ready** |
| Thesis draft: intro, RQs, methodology, related work | **Draft v1** |

### What's Missing
| Item | Experiments Needed |
|------|-------------------|
| Phase 2 LSTM seed 123 remaining folds | 2 fold-runs |
| Phase 2 LSTM seeds 456, 789, 1024 | 3 × 8 = 24 fold-runs |
| Phase 2 TCN (all 5 seeds × 8 folds) | 40 fold-runs |
| Phase 2 Transformer (all 5 seeds × 8 folds) | 40 fold-runs |
| Phase 2 GConvLSTM predefined_coarse (5 seeds × 8 folds) | 40 fold-runs |
| Phase 2 GConvLSTM predefined_fine (5 seeds × 8 folds) | 40 fold-runs |
| Phase 2 GConvLSTM adaptive (5 seeds × 8 folds) | 40 fold-runs |
| Phase 2 GConvGRU × 3 graph modes × 5 seeds × 8 folds | 120 fold-runs |
| Phase 2 A3TGCN × 3 graph modes × 5 seeds × 8 folds | 120 fold-runs |
| Phase 2 DCRNN × 3 graph modes × 5 seeds × 8 folds | 120 fold-runs |
| **Total Phase 2 remaining** | **~586 fold-runs** |

### Models × Graph Modes Matrix

| Model | Phase 1 BM | Phase 1 Epi | Phase 2 DSA (coarse) | Phase 2 DSA (fine) | Phase 2 DSA (adaptive) |
|-------|-----------|-------------|---------------------|-------------------|----------------------|
| LSTM | ✅ none | ✅ none | 🔄 none | — | — |
| TCN | ✅ none | ✅ none | ❌ none | — | — |
| Transformer | ✅ none | ✅ none | ❌ none | — | — |
| GConvLSTM | ✅ pred/adapt | ✅ pred/adapt | ❌ pred_coarse | ❌ pred_fine | ❌ adaptive |
| GConvGRU | ✅ pred/adapt | ✅ pred/adapt | ❌ pred_coarse | ❌ pred_fine | ❌ adaptive |
| A3TGCN | ✅ pred/adapt | ✅ pred/adapt | ❌ pred_coarse | ❌ pred_fine | ❌ adaptive |
| DCRNN | ✅ pred/adapt | ✅ pred/adapt | ❌ pred_coarse | ❌ pred_fine | ❌ adaptive |

---

## 1. ENVIRONMENT SETUP (Docker End-to-End)

### 1.1 Build Docker Image
```bash
cd /Users/timele23/Library/Mobile\ Documents/com~apple~CloudDocs/Thesis_Master
docker compose build
```

### 1.2 Verify Environment Inside Docker
```bash
docker compose run --rm thesis bash -c "
  python -c 'import torch; print(torch.__version__)' && \
  python -c 'import torch_geometric; print(torch_geometric.__version__)' && \
  python -c 'import torch_geometric_temporal; print(torch_geometric_temporal.__version__)' && \
  python -c 'from src.models.factory import ALL_MODELS; print(ALL_MODELS)' && \
  echo 'ALL IMPORTS OK'
"
```

### 1.3 Run Sanity Test
```bash
docker compose run --rm thesis python test_pipeline.py
```

### 1.4 Dry Run — See All Pending Experiments
```bash
docker compose run --rm thesis python -m scripts.run_all --dry_run
```

---

## 2. EXECUTION STRATEGY

### Priority Order (what to run and why)

Training on CPU is slow for Phase 2 (each LOSO fold ~ 3-10 min per epoch).
**Recommended**: use the Docker container for reproducibility verification, then run actual training on:
- **Local Mac (MPS)**: faster than CPU for small/medium models
- **Google Colab (GPU)**: critical for Phase 2 bulk runs

### Step 2.1: Complete Phase 2 Baselines First
Baselines provide the reference against which TGNNs are compared (RQ1 & RQ3).

```bash
# TCN baseline — all 5 seeds
for SEED in 42 123 456 789 1024; do
  python -m scripts.run_experiment --dataset DSA --model tcn --graph_mode none \
    --seed $SEED --phase 2 --device cpu
done

# Transformer baseline — all 5 seeds
for SEED in 42 123 456 789 1024; do
  python -m scripts.run_experiment --dataset DSA --model transformer --graph_mode none \
    --seed $SEED --phase 2 --device cpu
done

# Complete LSTM remaining seeds
for SEED in 456 789 1024; do
  python -m scripts.run_experiment --dataset DSA --model lstm --graph_mode none \
    --seed $SEED --phase 2 --device cpu
done
```

### Step 2.2: Run Phase 2 TGNNs with Predefined Coarse Graph
Most important graph mode — directly tests anatomical domain knowledge (RQ2 core).

```bash
for MODEL in gconv_lstm gconv_gru a3tgcn dcrnn; do
  for SEED in 42 123 456 789 1024; do
    python -m scripts.run_experiment --dataset DSA --model $MODEL \
      --graph_mode predefined_coarse --seed $SEED --phase 2 --device cpu
  done
done
```

### Step 2.3: Run Phase 2 TGNNs with Adaptive Graph
Tests data-driven graph discovery vs domain knowledge (RQ2 counterpart).

```bash
for MODEL in gconv_lstm gconv_gru a3tgcn dcrnn; do
  for SEED in 42 123 456 789 1024; do
    python -m scripts.run_experiment --dataset DSA --model $MODEL \
      --graph_mode adaptive --seed $SEED --phase 2 --device cpu
  done
done
```

### Step 2.4: Run Phase 2 TGNNs with Predefined Fine Graph
45-node fine-grained graph — most expensive but tests full channel-level graph.

```bash
for MODEL in gconv_lstm gconv_gru a3tgcn dcrnn; do
  for SEED in 42 123 456 789 1024; do
    python -m scripts.run_experiment --dataset DSA --model $MODEL \
      --graph_mode predefined_fine --seed $SEED --phase 2 --device cpu
  done
done
```

---

## 3. ANALYSIS PIPELINE

### Step 3.1: After Each Batch of Experiments
```bash
docker compose run --rm analyse
# OR
python -m scripts.analyse_results
```

This generates:
- `results/analysis/all_results.csv` — master comparison table
- `results/analysis/table_*.csv` and `table_*.tex` — per-dataset LaTeX tables
- `results/analysis/wilcoxon_*.csv` — pairwise significance tests
- `results/analysis/confusion_matrices/` — per-model confusion matrix plots
- `results/analysis/statistical_analysis.json` — bootstrap CIs, aggregated stats

### Step 3.2: Statistical Testing (Built Into analyse_results.py)
- **Wilcoxon signed-rank** test between model pairs (across seeds)
- **Bootstrap confidence intervals** for accuracy and F1
- **Per-fold variance** reporting for DSA LOSO

### Step 3.3: Computational Cost Comparison
Already tracked per experiment:
- `n_parameters` — trainable parameter count
- `inference_time_mean_ms` — mean inference latency
- `total_train_time_s` — wall-clock training time

---

## 4. TECHNIQUES FROM REFERENCES TO EXPLORE

Based on the references in `references.bib` and the literature on these exact datasets:

### 4.1 Already Implemented
| Technique | Reference | Status |
|-----------|-----------|--------|
| LSTM baseline | Hochreiter 1997 | ✅ |
| TCN with causal dilated conv | Bai 2018 | ✅ |
| Transformer with positional encoding | Vaswani 2017 | ✅ |
| GConvLSTM (Chebyshev spectral + LSTM) | Seo 2018 | ✅ |
| GConvGRU (Chebyshev spectral + GRU) | Seo 2018 | ✅ |
| A3TGCN (attention temporal GCN) | Bai 2021 | ✅ |
| DCRNN (diffusion conv + GRU) | Li 2018 | ✅ |
| Graph WaveNet-style adaptive adjacency | Wu 2019 | ✅ (in adaptive mode) |
| Domain-knowledge predefined graphs | Altun 2010 | ✅ |
| Attention-based adaptive graph | Custom | ✅ |

### 4.2 Potential Additions (for stronger results)
These could strengthen the benchmark — implement only if time permits:

| Technique | Why | Complexity |
|-----------|-----|-----------|
| **Bidirectional LSTM** | Already supported (`bidirectional: true`), just not run yet | Config change only |
| **Label smoothing** | Configured at 0.0, try 0.1 for DSA (19 classes) | Config change only |
| **Correlation-based graph** | Use Pearson correlation of training data as adjacency | Medium — add to `predefined.py` |
| **Multi-scale TCN** | Parallel TCN branches with different dilation rates | Medium |
| **GATv2 temporal** | Graph Attention Network v2 as spatial layer | Medium — extend `tgnn.py` |
| **MixHop / higher-order neighbors** | Multi-hop message passing | Medium |
| **InceptionTime** | State-of-art TSC baseline not GNN-based | New model file |

### 4.3 Hyperparameter Variations Worth Testing
| Parameter | Current | Try |
|-----------|---------|-----|
| LSTM `bidirectional` | false | true |
| `label_smoothing` | 0.0 | 0.05, 0.1 |
| Transformer `nhead` | 8 | 4 (for small datasets) |
| TGNN `hidden_dim` | 64 | 32, 128 |
| Adaptive `embed_dim` | 16/32 | 8, 64 |
| Learning rate | 0.001/0.0005 | 0.0003 |
| Cosine annealing `T_max` | 200 | match actual epochs |

---

## 5. THESIS WRITING MILESTONES

### Chapter Structure (Target)
1. **Introduction** — Problem, motivation, RQs *(draft done)*
2. **Related Work** — Sequence models, GNNs/TGNNs, HAR, benchmarking *(draft done, expand)*
3. **Methodology** — Models, graphs, training protocol, evaluation *(draft done, formalize)*
4. **Experimental Setup** — Datasets, splits, configuration, reproducibility
5. **Results & Discussion** — Phase 1, Phase 2, cross-phase comparison, RQ answers
6. **Conclusions** — Summary, limitations, future work

### What to Write When
| Section | Write After |
|---------|-------------|
| Results Phase 1 (complete tables, discussion) | Now — data is complete |
| Results Phase 2 baselines | After Step 2.1 |
| Results Phase 2 TGNNs | After Steps 2.2-2.4 |
| Cross-phase analysis (RQ3) | After all Phase 2 done |
| Statistical significance section | After analysis pipeline |
| Computational cost discussion | After all experiments |
| Conclusions | Last |

---

## 6. QUICK-START COMMANDS

```bash
# ── Build and verify ──
docker compose build
docker compose run --rm thesis python test_pipeline.py

# ── Preview all planned experiments ──
docker compose run --rm thesis python -m scripts.run_all --dry_run

# ── Run everything (Phase 1 will skip, Phase 2 continues) ──
docker compose run --rm thesis python -m scripts.run_all --device cpu

# ── Run just Phase 2 ──
docker compose run --rm phase2

# ── Run a single experiment interactively ──
docker compose run --rm thesis python -m scripts.run_experiment \
  --dataset DSA --model a3tgcn --graph_mode predefined_coarse \
  --seed 42 --phase 2 --device cpu

# ── Generate analysis after experiments ──
docker compose run --rm analyse

# ── Interactive shell for debugging ──
docker compose run --rm thesis
```

---

## 7. FILE ORGANIZATION

```
Thesis_Master/
├── configs/                    # YAML experiment configs
│   ├── default.yaml           # Shared defaults
│   ├── phase1.yaml            # BasicMotions + Epilepsy
│   └── phase2.yaml            # DSA
├── src/                        # Core library
│   ├── data/                  # Datasets, preprocessing, splits
│   ├── models/                # LSTM, TCN, Transformer, TGNN, factory
│   ├── graphs/                # Predefined + adaptive graph construction
│   ├── training/              # Trainer, losses
│   ├── evaluation/            # Metrics, statistical analysis
│   └── utils/                 # Seed, device, logging
├── scripts/                    # Experiment runners
│   ├── run_experiment.py      # Single experiment
│   ├── run_all.py             # Orchestrator
│   └── analyse_results.py     # Post-hoc analysis
├── results/                    # All experiment outputs
│   ├── phase1/                # ✅ Complete
│   ├── phase2/                # 🔄 In progress
│   └── analysis/              # Tables, plots, statistics
├── thesis_draft/               # LaTeX manuscript
├── Dockerfile                  # Full training image
├── docker-compose.yml          # Service definitions
├── PLAN.md                     # ← THIS FILE
└── requirements.txt            # Python dependencies
```

---

## 8. RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| Phase 2 too slow on CPU | Use MPS locally or Colab GPU; reduce `max_folds` for quick validation |
| TGNN overfits on small DSA folds | Early stopping (patience=25), dropout, gradient clipping all configured |
| Near-ceiling Phase 1 results | Already observed — DSA provides the discriminative benchmark |
| Docker build fails | Pin all versions; cached layers minimize rebuild time |
| Disk space (many results) | Each experiment ~100KB; 600 experiments ≈ 60MB total |
| **iCloud Drive file access** | Large .npy files get evicted; copy UCI data to /tmp before preprocessing |

---

## 9. iCloud WORKAROUND (CRITICAL)

The project lives on iCloud Drive. Large files (200MB+ DSA processed data) can be
evicted to cloud storage and cause `TimeoutError` / `Resource deadlock avoided`
when accessed from Python or Docker.

### First-Time Setup (after clone or eviction)
```bash
# 1. Copy raw UCI data to local tmp directory
cp -r UCI /tmp/thesis_data_UCI

# 2. Regenerate DSA processed data from local copy
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data.download import load_dsa_raw
from src.data.preprocessing import _preprocess_and_save_dsa
from pathlib import Path
# Load from local copy
X, y, subjects = load_dsa_raw(Path('/tmp/thesis_data_UCI'))
# Save to project processed dir
_preprocess_and_save_dsa(X, y, subjects)
"

# 3. Verify
python3 -c "
import numpy as np
X = np.load('data/processed/DSA/X.npy')
print(f'DSA OK: {X.shape}')
"
```

### Permanent Fix
Consider symlinking the project to a non-iCloud local directory:
```bash
# Move to local disk
mv ~/Library/Mobile\ Documents/com~apple~CloudDocs/Thesis_Master ~/Thesis_Master
# Symlink back if needed
ln -s ~/Thesis_Master ~/Library/Mobile\ Documents/com~apple~CloudDocs/Thesis_Master
```
