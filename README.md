# Temporal Graph Neural Networks for Multivariate Time Series Classification: A Systematic Benchmark

## Master's Thesis – MSc Artificial Intelligence

### Overview

This repository contains the complete codebase for a systematic benchmark study comparing **Temporal Graph Neural Networks (TGNNs)** against established sequence models (**LSTM**, **TCN**, **Transformer**) for multivariate time series classification, with a focus on physical activity recognition from body-worn inertial sensors.

### Research Questions

1. Do TGNNs consistently outperform established sequence models (LSTM, TCN, Transformer) for multivariate time series classification?
2. Does the graph construction strategy (predefined domain-knowledge adjacency vs. adaptive learned adjacency) critically determine the TGNN advantage?
3. How does the benefit of explicit relational modelling change with increasing dimensionality and structural complexity?

### Datasets

| Dataset | Phase | Variables | Instances | Classes | Source |
|---------|-------|-----------|-----------|---------|--------|
| BasicMotions | 1 | 6 | 80 | 4 | UEA Archive |
| Epilepsy | 1 | 3 | 275 | 4 | UEA Archive |
| Daily & Sports Activities | 2 | 45 | 9120 | 19 | UCI Repository |

### Models

**Baselines:**
- LSTM (Long Short-Term Memory)
- TCN (Temporal Convolutional Network)
- Transformer

**Temporal Graph Neural Networks:**
- GConvLSTM (Graph Convolutional LSTM)
- GConvGRU (Graph Convolutional GRU)
- A3TGCN (Attention Temporal Graph Convolutional Network)
- DCRNN (Diffusion Convolutional Recurrent Neural Network)
- ASTGCN (Attention-based Spatial-Temporal Graph Convolutional Network)

### Project Structure

```
Thesis_Master/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   ├── phase1.yaml           # Phase 1 experiment configs
│   └── phase2.yaml           # Phase 2 experiment configs
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py       # Dataset downloaders
│   │   ├── preprocessing.py  # Preprocessing pipelines
│   │   ├── datasets.py       # PyTorch Dataset classes
│   │   └── splits.py         # Train/val/test splitting
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── predefined.py     # Predefined graph construction
│   │   └── adaptive.py       # Adaptive/learned graph construction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py           # LSTM baseline
│   │   ├── tcn.py            # TCN baseline
│   │   ├── transformer.py    # Transformer baseline
│   │   ├── tgnn.py           # TGNN architectures
│   │   └── factory.py        # Model factory
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   └── losses.py         # Loss functions
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── analysis.py       # Statistical analysis
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py  # Logging utilities
│       ├── seed.py           # Reproducibility
│       └── device.py         # Device management
├── scripts/
│   ├── run_experiment.py     # Main experiment runner
│   ├── run_all.py            # Run all experiments
│   └── analyse_results.py   # Results analysis & plots
├── notebooks/
│   └── exploration.ipynb     # Data exploration
├── results/                  # Experiment outputs
├── figures/                  # Generated figures
└── data/                     # Downloaded datasets
    ├── raw/
    └── processed/
```

### Setup

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Download datasets
python -m src.data.download --all

# Run a single experiment
python scripts/run_experiment.py --config configs/phase1.yaml --model lstm --dataset BasicMotions

# Run all experiments
python scripts/run_all.py

# Generate analysis and figures
python scripts/analyse_results.py
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- PyTorch Geometric Temporal
- scikit-learn
- pandas, numpy, scipy
- matplotlib, seaborn
- PyYAML
- tqdm
