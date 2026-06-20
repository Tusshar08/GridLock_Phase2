# Model 1 v2 — Changelog & Leakage Refinement Summary

## File Created
- `notebooks/model1/model_road_closure_v2.ipynb` (45 cells, 16 sections)
- Outputs to: `outputs/model_road_closure_v2/`

---

## Leakage Fixes Applied

### 1. Separate Calibration Split (Critical Fix)
- **v1 Problem**: Sigmoid calibration was fitted on the validation set, then the same validation set was used to evaluate calibration quality and select ensemble weights. This is **double-dipping** — the calibrator has already seen the val labels, inflating reported Brier scores.
- **v2 Fix**: Introduced a 4-way chronological split:
  - **Train (56%)** — model training only
  - **Calibration (14%)** — carved from training data, used exclusively for Platt scaling
  - **Validation (15%)** — ensemble weight search + threshold optimization (never touched by calibrator)
  - **Test (15%)** — final held-out evaluation

### 2. Ensemble Weight Selection on Clean Validation
- **v1 Problem**: Ensemble weights (LGB vs XGB) were searched on the same validation set that calibration was fitted on.
- **v2 Fix**: Weight grid search runs on the validation set which the calibrator has **never seen**, giving honest PR-AUC estimates for weight selection.

### 3. Consistent Preprocessor in Forward-Chaining Handoff (Section 16)
- **v1 Problem**: The Model 2 handoff loop created a **new `ColumnTransformer` per fold**, causing OneHotEncoder category drift between folds and inconsistent feature spaces vs. the main model.
- **v2 Fix**: All forward-chaining folds reuse the **same preprocessor** fitted once on training data (Section 4). `transform_features()` is called uniformly, ensuring identical encoded feature columns across the main model and every handoff fold.

### 4. Project-Relative Paths
- **v1 Problem**: Hardcoded Windows paths (`D:\Python\Gridlock\Phase 2\theme 2\...`) made the notebook non-portable.
- **v2 Fix**: Auto-discovers `PROJECT_ROOT` by walking parent directories for `outputs/feature_engineering_v1/`. All paths are `pathlib.Path`-based.

---

## Structural Changes

### Data Source
| Item | v1 | v2 |
|------|----|----|
| Feature data | `outputs/feature_engineering_v1/road_closure_features_v1.csv` | Same |
| Duration base | `outputs/feature_engineering_v1/duration_base_features_v1.csv` | Same |
| Output dir | `outputs/model_road_closure/` | `outputs/model_road_closure_v2/` |

### Split Design
| Split | v1 | v2 |
|-------|----|----|
| Train | 70% | 56% |
| Calibration | _(none — used val set)_ | 14% (carved from training) |
| Validation | 15% | 15% |
| Test | 15% | 15% |

### Notebook Sections (16 total)
1. **Import Libraries** — added `re`, `seaborn`, `BaseEstimator`/`ClassifierMixin`, `optuna`
2. **Load Feature-Engineered Dataset** — project-relative path discovery
3. **Leakage Audit** — blocked features check, QA column check, chronological order check, statistical AUC/correlation diagnostics on training portion only
4. **Data Preparation & Chronological Split** — 4-way split with separate calibration carve-out
5. **Baseline Model (Logistic Regression)** — class-weighted LR as sanity baseline
6. **LightGBM Model** — default-param run with early stopping against calibration set
7. **XGBoost Model** — default-param run with early stopping against calibration set
8. **Hyperparameter Optimization + Ensemble Training** — 30 Optuna trials each for LGB and XGB, final models calibrated on calibration split, ensemble weights selected on validation
9. **Calibration Review** — comparison table + calibration curve + probability distribution plots
10. **Threshold Optimization** — F1-plateau method on validation set
11. **Test Set Evaluation** — classification report, confusion matrix, ROC-AUC, PR-AUC, Brier
12. **Feature Importance Analysis** — dual LGB/XGB gain importance bar charts
13. **SHAP Explainability** — TreeExplainer summary plot + waterfall for high/low risk events
14. **Save Final Model & Artifacts** — inference bundle `.pkl`, predictions `.csv`, ensemble importance, threshold metadata
15. **Production-Ready Prediction Function** — `predict_road_closure_probability()` with risk level and action
16. **Leakage-Safe Model 2 Handoff** — forward-chaining with consistent preprocessor, per-fold calibration, fallback to `past_closure_global_rate` for warm-up rows

---

## Output Artifacts

| File | Description |
|------|-------------|
| `model1_inference_bundle.pkl` | Portable model bundle containing LGB booster, XGB booster, calibrators, preprocessor, feature cols, threshold, weights |
| `model1_road_closure_predictions.csv` | Test set predictions with metadata, component probabilities, ensemble probability, percentage |
| `model2_duration_handoff.csv` | Duration-band model input with forward-chained road closure probabilities (no in-sample leakage) |

---

## Key Design Decisions

- **Early stopping uses calibration set, NOT validation**: Both LGB and XGB early-stop against `X_cal/y_cal`. This keeps `X_val/y_val` fully independent for ensemble tuning.
- **Optuna objectives evaluate on validation**: Each trial's score is `average_precision_score(y_val, preds)`, ensuring hyperparameter search doesn't leak through early-stopping data.
- **Feature name sanitization**: `make_safe_unique_names()` ensures XGBoost-compatible feature names (no special characters) with uniqueness guarantees.
- **Forward-chaining handoff uses best Optuna params**: The handoff loop applies the same `best_params_lgb` found in Section 8, rather than separate hardcoded params.
- **Warm-up fallback**: First 20% of rows use `past_closure_global_rate` (a strict past-only statistic) as the road closure probability, since there isn't enough history to train a stable model.

---

## Dependencies
No new dependencies added. Uses the same stack as v1:
- `lightgbm`, `xgboost`, `optuna`, `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`
