# Auxetic re-entrant honeycomb interlayers in FDM-PETG single-lap joints: data and code

Data and analysis code accompanying the manuscript *"Auxetic Re-entrant Honeycomb Interlayers in FDM-PETG Single-Lap Adhesive Joints: An Experimental and Machine-Learning-Guided Design"* (M. A. Chowdhury, S. Chowdhury, Md. A. Islam, Md. G. Kader; Department of Mechanical Engineering, Khulna University of Engineering & Technology).

Everything in `results/` is regenerated from one file, `data/specimens.csv`, by the scripts in `code/`.

## Contents

```
data/
  specimens.csv                  master table, 176 specimens x 32 columns
  SLJ_Auxetic_Repository.xlsx    same table + configuration summaries, data dictionary, all derived tables
  Database_Repository.docx       printable description of the workbook
  README_data.md                 short column notes
code/
  common.py                      data loading, model definitions, hyper-parameter grids
  01_descriptive_and_validation_stats.py
  02_cross_validation.py
  03_shap_and_input_ablation.py
  04_bayesian_optimization.py
  05_failure_mode_statistics.py
  06_configuration_level_statistics.py
  07_check_reported_values.py
  08_extra_models.py
results/                         outputs of the scripts (CSV / JSON), as generated for the manuscript
run_all.sh                       runs 01-08 in order
requirements.txt
LICENSE
```

## The data

160 training specimens (40 configurations x 4 replicates: Base, three uniform auxetic families at 45/60/75 deg, two graded families, Flat, Spacer) and 16 validation specimens (4 candidates x 4 replicates) chosen by the Bayesian-optimization loop. Per specimen: geometry (re-entrant angle, strut thickness t, interlayer thickness IL, auxetic index AI, relative density, porosity, grading ratio GR, cell aspect ratio CAR), measured epoxy thickness (training only), peak force Fmax, displacement at peak, E_tri = 0.5 * Fmax * delta_peak, and ASTM D5573 failure-mode label. Training rows also carry the cross-validation fold, out-of-fold predictions of five surrogates and the blend, and exact Shapley values of the GPR. Column definitions are on the `Dictionary` sheet of the workbook.

## Run order

```bash
pip install -r requirements.txt
bash run_all.sh            # about 12 minutes on 4 cores
```

or run the scripts one by one from `code/` (script 02 must run before 03 and 04):

| Script | What it does | Main outputs in `results/` | Manuscript |
|---|---|---|---|
| `01_descriptive_and_validation_stats.py` | Family means vs Base, Welch tests with Holm correction, Hedges' g, ANOVA / Kruskal-Wallis / Tukey, validation candidates vs Base, best training configuration and 45 deg family | `descriptive_stats.csv`, `validation_tests.csv` | Family comparison and validation results |
| `02_cross_validation.py` | Grouped 5-fold CV (folds split by configuration) for XGB, RF, GPR, SVR, Ridge and the XGB/GPR blend; hyper-parameter search (12 random draws, inner GroupKFold(3)); final fits | `cv_summary.csv`, `cv_metrics_per_fold.csv`, `cv_hyperparameters.csv`, `cv_oof_predictions.csv` | Section 4.3, Table S1b |
| `03_shap_and_input_ablation.py` | Exact Shapley values of the GPR (final fit and per fold), ARD length-scales, GPR feature-removal CV, input correlation matrix | `shap_gpr_importance.csv`, `shap_gpr_final.csv`, `shap_gpr_per_fold.csv`, `gpr_ablation.csv`, `gpr_ard_length_scales.csv` | Section 4.3, Tables S3, S5, S6 |
| `04_bayesian_optimization.py` | Expected-improvement sweep over 41 x 41 x 11 (theta, IL, t) for nine blend weights; retrospective predictions for the four validation candidates | `ei_sweep.csv`, `ei_surface_top20.csv`, `retrospective_predictions.csv` | Section 5.1, Tables S2, S4 |
| `05_failure_mode_statistics.py` | Chi-square, Monte Carlo permutation tests (specimen and configuration level, 1e5 permutations), Wilson intervals, Fisher exact tests | `failure_mode_stats.json`, `failure_mode_tables.csv` | Table S7 |
| `06_configuration_level_statistics.py` | Configuration-level correlations with cluster bootstrap, OLS vs random-intercept mixed model, 45 deg vs Base, two-way ANOVA on the 27 uniform configurations | `config_level_stats.json`, `config_anova.json` | Tables S8, S8b |
| `07_check_reported_values.py` | Compares ten headline numbers with the manuscript | console output | - |
| `08_extra_models.py` | Ten additional benchmark regressors (CatBoost, LightGBM, Gradient Boosting, AdaBoost, Extra Trees, Decision Tree, KNN, Kernel Ridge, MLP, ElasticNet) under the same grouped 5-fold CV and tuning protocol; mean abs. SHAP of each on a final fit. Run after the main analysis (exploratory); needs `lightgbm` and `catboost` | `extra_models_cv_summary.csv`, `extra_models_shap.csv` | Table S16 |

## Modelling details

* Inputs (7): AI, porosity, IL, epoxy thickness, t, GR, CAR. Target: Fmax.
* Models: XGBoost, Random Forest, GPR (StandardScaler, constant x Matern-5/2 ARD + white-noise kernel, `normalize_y`, 3 restarts), RBF SVR (epsilon = 0.03), Ridge. Seed 42 throughout.
* Blend: w * XGB + (1 - w) * GPR with w proportional to the inverse mean fold-RMSE (0.43 / 0.57).
* Acquisition: expected improvement with F+ = 2132.3 N (best training configuration mean) and a 1 N margin. Candidate rows use epoxy 0.30 mm, GR = CAR = 1; porosity is the measured value in the retrospective check and clip(AI / 0.414 x 82, 0, 95) in the sweep.
* Shapley values: `shap.explainers.Exact` with an independent masker, all training rows as background.

## Notes on reproducibility

* The scripts reproduce the reported values to the precision printed in the manuscript (`07_check_reported_values.py`). Results were generated with Python 3.11, scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0, statsmodels 0.15.0, SciPy 1.17.1, NumPy 2.4.4, pandas 3.0.2; other versions may differ in the last digits, mainly in hyper-parameter choices of the tuned models.
* The four selection-time predictions (2058, 2071, 2095, 2089 N) were recorded when the candidates were chosen, before the validation specimens were tested. They are listed in the workbook (`Validation_Predictions`) but cannot be regenerated from the final data; the retrospective predictions in `retrospective_predictions.csv` use the final models.
* Validation specimens were never used for training or tuning.

## License

MIT (see `LICENSE`).

## Citation

Archived release with DOI: to be added on acceptance.
