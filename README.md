# ForecastX-AI

A production-grade, multi-model sales forecasting system for omnichannel retail data. The pipeline combines **LightGBM**, **Temporal Fusion Transformer (TFT)**, and **GARCH-X** into a Ridge meta-learner to produce point forecasts and confidence-band outputs across Q1, H1, and Annual horizons.

---

## Overview

This project tackles revenue forecasting across a large omnichannel dataset (~1M orders, Jan 2010 – Sep 2020) spanning multiple regions, sales channels, and product categories. Rather than a single monolithic model, it uses an ensemble stack where each component contributes a distinct modelling strength.

| Component | Role |
|---|---|
| LightGBM | Fast, interpretable direct multi-step forecasting with lag + calendar features |
| Temporal Fusion Transformer | Sequential pattern learning with attention-based variable importance |
| GARCH-X | Volatility modelling for calibrated P10/P50/P90 confidence bands |
| Ridge meta-learner | Combines LightGBM + TFT predictions into a single blended forecast |

---

## Architecture

```
Raw orders (1M rows)
        │
        ▼
Monthly aggregation per (Region × Sales Channel × Item Type)
        │
        ▼
Feature engineering
  ├── Lag features: 1m, 2m, 3m, 6m, 12m
  ├── Rolling stats: mean & std over 3m, 6m, 12m windows
  ├── Calendar: month/quarter sin-cos encodings
  ├── YoY & MoM growth rates
  └── Channel share
        │
   ┌────┴────┐
   ▼         ▼
LightGBM   TFT (24-month encoder window)
   │         │
   └────┬────┘
        ▼
  Ridge meta-learner
        │
        ▼
  GARCH-X confidence bands
        │
        ▼
Output: P10 / P50 / P90 forecasts per group × horizon
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [1 M Sales Records – Kaggle]((https://www.kaggle.com/datasets/kalyugvasi/sale-dataset)) |
| Rows | 1,048,575 orders |
| Date range | Jan 2010 – Sep 2020 |
| Granularity | Monthly per (channel × category) after aggregation |
| Target | `Total Revenue` |
| Leaky columns excluded | `Units Sold`, `Unit Price`, `Unit Cost`, `Total Cost`, `Total Profit` |

---

## Features

- Chronological 80/20 train/validation split — no data leakage
- Per-group lag and rolling features with 12-month warmup
- Cyclical calendar encodings (sin/cos) for month and quarter
- SHAP-based feature importance for LightGBM
- TFT attention weight visualisation
- GARCH volatility forecast as confidence-band calibrator
- Multi-audience output layer: demand planners, FP&A, and executive roll-ups

---

## Requirements

```
python >= 3.9
lightgbm
pytorch-forecasting
pytorch-lightning
arch
shap
scikit-learn
pandas
numpy
matplotlib
seaborn
torch
```

Install all dependencies:

```bash
pip install lightgbm pytorch-forecasting pytorch-lightning arch shap scikit-learn pandas numpy matplotlib seaborn torch
```

---

## Usage

1. Place `Sales_Records.csv` in the project root.
2. Open `ForecastX-AI.ipynb` in Jupyter.
3. Run all cells top to bottom. Each step is labelled and self-contained.

The final output (Step 13–15) produces a forecast table with P10/P50/P90 bands and directional signals (▲ Up / → Flat / ▼ Down) for every region × item type × horizon combination.

---

## Results

Evaluation metrics reported on the held-out validation set (~2 years):

| Horizon | Model | WAPE |
|---|---|---|
| Q1 (3 months) | LightGBM | — |
| Q1 | Meta (LGB + TFT) | — |
| H1 (6 months) | LightGBM | — |
| Annual (12 months) | LightGBM | — |


---

## Project Structure

```
.
├── ForecastX-AI.ipynb   # Main notebook
├── Sales_Records.csv                      # Dataset (not included)
├── README.md
└── LICENSE
```

---

## Author

**Saurabh Kesarwani** — [@kesarwani2509](https://github.com/kesarwani2509)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
