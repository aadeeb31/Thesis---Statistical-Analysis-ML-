# Auxetic-Clamping SLJ: ML-Guided Closed-Loop Design (v10.0)

Code accompanying the manuscript:

> Chowdhury, M. A., Chowdhury, S., Islam, M. A., & Kader, M. G. **"Auxetic-Clamping Suppression of Peel-Stress Failure in FDM-PETG Single-Lap Adhesive Joints: A Closed-Loop, Machine-Learning-Guided Design Study."** *Manuscript submitted for peer review, 2026.*

**This manuscript has not yet been published.** The citation above (and in `CITATION.cff` / `LICENSE`) is a placeholder — once the paper is accepted, update `CITATION.cff`'s `preferred-citation` block to an `article` type with the journal, volume, pages, and DOI, and update the `LICENSE` and this README to match. See the "Citation" section below.

This repository contains a single, heavily-commented analysis script (`SLJ_Analysis_Pro.py`) and its notebook version (`SLJ_Analysis_Pro.ipynb`) that reproduce the full statistical analysis, machine-learning pipeline, Bayesian-Optimization inverse design, and physical-validation comparison described in the manuscript, from a 176-specimen experimental dataset (160 training + 16 physical-validation specimens).

## Contents

| File | Description |
|---|---|
| `SLJ_Analysis_Pro.ipynb` | The main deliverable — the full pipeline split into markdown+code "story" cells (see notebook for the section-by-section walkthrough). |
| `SLJ_Analysis_Pro.py` | The same pipeline as a single script (useful for batch/Colab runs or diffing against the original). |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | Custom MIT-style license with a mandatory citation clause. |
| `CITATION.cff` | Machine-readable citation (GitHub's "Cite this repository" button uses this). |

## Data

The dataset (`SLJ_ML_Database_v7.8.xlsx` or equivalent, sheet name `ML_Dataset`) is **not included** in this repository — place your copy in the same folder as the notebook/script before running. Per the manuscript, the full specimen-level dataset is openly available via OSF: **https://doi.org/10.17605/OSF.IO/FTZCP**.

Expected columns (see the "Loading the dataset" cell in the notebook for the exact schema): `specimen_id`, `configuration`, `topology_type`, `auxetic_index`, `strut_mm`, `interlayer_mm`, `porosity_pct`, `grading_ratio`, `epoxy_thickness_mm`, `cell_aspect_ratio`, `l_abs_mm`, `cell_pitch_x_mm`, `max_force_N`, `energy_Nmm`, `displacement_mm`, `stiffness_N_mm`, `failure_mode`.

**Before running against your real spreadsheet**, check two assumptions flagged inline in the code:

1. **Validation-specimen naming.** The four out-of-grid Bayesian-Optimization candidates (V-α/β/δ/ε, Table 5) are matched out of the `configuration` column with a flexible regex (`V_alpha`/`V-alpha`/`V-α`, case-insensitive). The exact string your spreadsheet uses wasn't available when this notebook was prepared — check the printed "N training / N validation specimens" line against your expectations, and adjust `_val_mask` near the top of the script if it doesn't find all 16.
2. **Hyperparameter search ranges.** The `PARAM_DISTRIBUTIONS` used for the nested `RandomizedSearchCV` tuning (Sec. 4.1) are reasonable literature-typical stand-ins for each model class, since the manuscript's exact ranges live in Supplementary Table S1, which wasn't available either. Swap in the real ranges if you have that file — the tuning *procedure* (nested search + across-fold-mode selection) already matches Sec. 4.1 regardless.

## Requirements

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook SLJ_Analysis_Pro.ipynb
```

Runtime knobs (useful for a fast smoke-test before committing to the full run, which can take 10-30+ minutes depending on your machine):

```bash
SLJ_N_BOOTSTRAP=200 SLJ_N_ITER_SEARCH=4 SLJ_INNER_CV_FOLDS=2 jupyter nbconvert --to notebook --execute SLJ_Analysis_Pro.ipynb
```

(Defaults: `SLJ_N_BOOTSTRAP=10000`, `SLJ_N_ITER_SEARCH=15`, `SLJ_INNER_CV_FOLDS=3` — matching the manuscript's stated bootstrap resamples and search budget.)

## What this code does

- **Section 2.6 / 3.1-3.2 statistics**: an assumption-checked decision tree (Shapiro-Wilk -> Levene -> ANOVA / Kruskal-Wallis / Welch's F-test), Tukey HSD with compact-letter display, Cohen's d with Hedges-Olkin CIs, a chi-square/Cramér's V failure-mode association test, a two-way ANOVA on (cell angle, interlayer thickness), and a 10,000-resample bootstrap for the AI activation-threshold confidence interval.
- **Section 4.1-4.3 ML pipeline**: five tuned surrogates (XGBoost, Random Forest, Gaussian Process Regression, SVR, Ridge) predicting peak force (Fmax) only, evaluated with configuration-grouped 5-fold cross-validation, explained with SHAP (XGBoost + Random Forest), and combined into an inverse-CV-RMSE-weighted ensemble.
- **Section 4.2 Bayesian Optimization**: an Expected-Improvement search over (angle, interlayer, strut thickness) under a strict 0.40 mm printability floor, with a geometric diversity filter for the final candidate ranking.
- **Section 4.5 / 5.1 physical validation** *(new in this rewrite — see below)*: scores the four physically-tested, out-of-grid candidates with the already-fitted ensemble and reports the signed surrogate optimism gap against measured results.

## Paper-vs-code coverage report

The original script this repository is based on covered most of the manuscript's core analyses, but a careful section-by-section comparison against the manuscript turned up several gaps and one correctness issue, all addressed in this v10.0 rewrite. Nothing described in the paper was left out, and every analysis in the original script was kept (either as a paper-matching computation or clearly labeled as bonus content).

### Fixed (present in the original code, but not matching the paper's stated method)

| Item | Original code | Paper (Sec.) | Fix |
|---|---|---|---|
| 5-fold cross-validation | Plain shuffled `KFold` | Configuration-grouped 5-fold, replicates never split across train/test (4.1) | Switched to `GroupKFold` on `configuration` |
| Bayesian-Optimization search range | AI gridded directly on `[0.035, 0.420]` — truncates below the paper's own best candidates (V-δ reaches AI=0.62) | θ ∈ [35°, 75°] via the Masters-Evans mapping (4.2) | Re-parameterized the search by θ, using the `ai_of_theta`/`theta_of_ai` helpers |
| Bayesian-Optimization printability constraint | None | Strict t ≥ 0.40 mm floor (4.2) | Added the printability filter before ranking candidates |
| Bayesian-Optimization scoring | GPR-only prediction | XGBoost/GPR convex ensemble (weights inverse to CV RMSE), GPR σ alone for EI uncertainty (4.1-4.2) | Rebuilt scoring to use the ensemble mean + GPR-only uncertainty |
| Bayesian-Optimization candidate ranking | Simple top-K by EI | Constant-liar-style batching with a geometric diversity filter (Δθ=5°, ΔIL=0.10mm, Δt=0.10mm) (4.2) | Added `select_diverse_top()` |

### Added (described in the paper, missing from the original code)

| Item | Paper (Sec.) |
|---|---|
| Shapiro-Wilk / Levene assumption checks with conditional Kruskal-Wallis / Welch's ANOVA fallback | 2.6, 3.1 |
| Two-way ANOVA on (cell angle, interlayer thickness) with interaction term, plus per-angle one-way ANOVA on interlayer thickness | 3.2 |
| 10,000-resample bootstrap CI for the AI activation threshold, and the paper's own Fig. 5 (AI-threshold scatter) | 2.6, 3.1 |
| `RandomizedSearchCV` hyperparameter tuning (across-fold mode) for all five surrogates, replacing hardcoded hyperparameters | 4.1 |
| Train-vs-CV R² overfitting-gap report | 4.3 |
| Random Forest SHAP (`TreeExplainer`) alongside XGBoost SHAP, and the paper-style attribution-robustness figure (Fig. 9a/b) | 4.3 |
| **An entire new Part I: Physical Validation** — scoring the four out-of-grid validation candidates (V-α/β/δ/ε) with the trained ensemble, computing the signed surrogate optimism gap against measured Fmax (Table 5), and reproducing Fig. 11 (predicted-vs-measured Fmax and Etri, validation failure-mode distribution) and Fig. 12(a) (full 176-specimen Fmax-vs-AI scatter with validation highlighted) | 4.5, 5.1 |

Note: Fig. 12(b) in the manuscript is a macro-photograph of the physical V-δ specimen cross-section — this is a physical photograph, not something a code artifact can reproduce, and is not included here.

### Kept as extra (present in the original code, not discussed in the paper)

These remain in the notebook, clearly labeled inline as bonus analyses beyond the paper's own scope — nothing was deleted:

- Synthetic representative force-displacement curves (Fig. 1) — a stylized illustrative aid, not a measured curve; the paper's own Fig. 1 is a schematic/geometry figure.
- ML modeling of `energy_Nmm` / `SEA_Nmm_mm3` / `stiffness_N_mm` in addition to Fmax — the paper models Fmax only (Sec. 4.1).
- The failure-mode classification pipeline (confusion matrix figure) — the paper doesn't model failure mode as an ML target.
- Leave-One-Group-Out cross-validation — a stricter stress test than the paper's own "5-fold" (GroupKFold) metric.

## Citation

If you use this code (or any part of it) in your own work, **you must cite the manuscript above**. A ready-to-use citation is provided in [`CITATION.cff`](CITATION.cff) (GitHub will show a "Cite this repository" button using this file). Since the manuscript is still under review, cite it for now as:

```
Chowdhury, M. A., Chowdhury, S., Islam, M. A., & Kader, M. G. (2026). Auxetic-Clamping
Suppression of Peel-Stress Failure in FDM-PETG Single-Lap Adhesive Joints: A Closed-Loop,
Machine-Learning-Guided Design Study. Manuscript submitted for peer review.
```

Once the paper is accepted and assigned a DOI, please update `CITATION.cff`, `LICENSE`, and this README with the full published citation (journal, volume, pages, DOI) — that's a small manual edit any time after acceptance.

## License

This code is released under a custom MIT-based license that **requires citation of the above manuscript for any use** — see [`LICENSE`](LICENSE).

## Authors

Mostofa Aadeeb Chowdhury, Sanjana Chowdhury, Md. Ashraful Islam*, Md. Golam Kader
Department of Mechanical Engineering, Khulna University of Engineering & Technology, Khulna-9203, Bangladesh
*Corresponding author
