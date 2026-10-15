"""Exact Shapley values of the GPR surrogate and GPR input-ablation CV.
Outputs: results/shap_gpr_final.csv (per specimen), shap_gpr_importance.csv, shap_gpr_per_fold.csv,
         gpr_ablation.csv, gpr_ard_length_scales.csv"""
import joblib, shap
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from common import *

s, tr, va = load()
X, y, g = tr[FEATURES].values, tr.Fmax_N.values, tr.configuration.values
folds = list(GroupKFold(5).split(X, y, g))


def exact_shap(model, background, Xexplain):
    ex = shap.explainers.Exact(model.predict, shap.maskers.Independent(background, max_samples=len(background)))
    return ex(Xexplain).values


# (a) final GPR, all 160 specimens as background and explained set
M = joblib.load(RES / "final_models.joblib")
gpr = M["models"]["GPR"]
sv = exact_shap(gpr, X, X)
imp = np.abs(sv).mean(0)
pd.DataFrame(sv, columns=[f"SHAP_GPR_{f}_N" for f in SHORT]).assign(specimen_id=tr.specimen_id.values).round(5)\
    .to_csv(RES / "shap_gpr_final.csv", index=False)
I = pd.DataFrame(dict(input=SHORT, mean_abs_SHAP_N=imp)).sort_values("mean_abs_SHAP_N", ascending=False)
print(I.round(1).to_string(index=False))
ls = gpr.named_steps["gpr"].kernel_.k1.k2.length_scale
pd.DataFrame(dict(input=SHORT, ARD_length_scale=ls)).to_csv(RES / "gpr_ard_length_scales.csv", index=False)
I.round(3).to_csv(RES / "shap_gpr_importance.csv", index=False)

# (b) stability across the five outer folds (model refit on the training part of each fold)
rows = []
for k, (a, b) in enumerate(folds, 1):
    m = make_model("GPR").fit(X[a], y[a])
    v = np.abs(exact_shap(m, X[a], X[b])).mean(0)
    rows.append(dict(fold=k, **{f"meanabsSHAP_{f}_N": x for f, x in zip(SHORT, v)}))
F = pd.DataFrame(rows)
F.round(3).to_csv(RES / "shap_gpr_per_fold.csv", index=False)
print(F.round(1).to_string(index=False))

# (c) GPR feature-removal CV (grouped 5-fold)
def cv_subset(cols):
    Xs = tr[cols].values; r = []
    for a, b in folds:
        m = make_gpr(len(cols)).fit(Xs[a], y[a]); r.append(r2_score(y[b], m.predict(Xs[b])))
    return np.mean(r), np.std(r, ddof=1)
rows = [dict(variant="all seven inputs", R2_mean=cv_subset(FEATURES)[0], R2_SD=cv_subset(FEATURES)[1])]
for f, sh in zip(FEATURES, SHORT):
    r = cv_subset([c for c in FEATURES if c != f]); rows.append(dict(variant=f"drop {sh}", R2_mean=r[0], R2_SD=r[1]))
for name, cols in [("drop t, porosity, epoxy", ["AI", "interlayer_IL_mm", "GR", "CAR"]),
                   ("drop AI and porosity", ["interlayer_IL_mm", "epoxy_thickness_mm", "strut_t_mm", "GR", "CAR"]),
                   ("only AI, IL, CAR", ["AI", "interlayer_IL_mm", "CAR"])]:
    r = cv_subset(cols); rows.append(dict(variant=name, R2_mean=r[0], R2_SD=r[1]))
A = pd.DataFrame(rows); A.round(4).to_csv(RES / "gpr_ablation.csv", index=False)
print(A.round(3).to_string(index=False))
print("\ninput correlation matrix\n", tr[FEATURES].corr().round(2).to_string())
