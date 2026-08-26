# Auxetic-Clamping SLJ: ML-Guided Closed-Loop Design (v10.0)

Code accompanying the manuscript:

> Chowdhury, M. A., Chowdhury, S., Islam, M. A., & Kader, M. G. **"Auxetic-Clamping Suppression of Peel-Stress Failure in FDM-PETG Single-Lap Adhesive Joints: A Closed-Loop, Machine-Learning-Guided Design Study."** *Manuscript submitted for peer review, 2026.*

**This manuscript has not yet been published.** The citation above (and in `CITATION.cff` / `LICENSE`) is a placeholder. Once the paper is accepted, `CITATION.cff`'s `preferred-citation` block will be updated to an `article` type with the journal, volume, pages, and DOI, and update the `LICENSE` and this README to match. See the "Citation" section below.

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

The dataset (`SLJ_ML_Database_v7.8.xlsx` or equivalent, sheet name `ML_Dataset`) is **not included** in this repository. The full specimen-level dataset is openly available via OSF: **https://doi.org/10.17605/OSF.IO/FTZCP**.

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
