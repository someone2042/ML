# Bank Marketing Machine Learning Pipeline

This repository contains Python scripts and machine learning models built to analyze and predict customer responses for bank direct marketing campaigns.

## 📁 Repository Structure

- `train_god_mode.py` - Advanced pipeline featuring extensive feature engineering, optuna hyperparameter tuning, multi-model training (LightGBM, XGBoost, CatBoost), and stacking.
- `train_stacking.py` - Stacking ensemble model implementation.
- `train_ultimate.py` - Core baseline and optimized model trainer.
- `train_extreme.py` - High-capacity classifier experiment script.
- `train_models.py` - Initial model testing and evaluation utilities.
- `REPORT.md` - Technical project report and benchmark metrics.
- `bank.csv` / `bank-full.csv` - Direct marketing dataset.

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Create and activate your virtual environment:

```bash
python -m venv env
# On Windows (PowerShell):
.\env\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```


### 2. Running Training Pipelines

To run the full optimized machine learning pipeline:

```bash
python train_god_mode.py
```

To run individual training scripts:

```bash
python train_ultimate.py
python train_stacking.py
```

## 📊 Results & Artifacts
Execution logs and models are stored in the `results/` folder, and CatBoost run logs are saved in `catboost_info/`. Refer to [`REPORT.md`](file:///c:/Users/MOHAMED/Desktop/Nouveau%20dossier/ML/REPORT.md) for detailed evaluation metrics.
