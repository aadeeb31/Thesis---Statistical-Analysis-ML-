"""Grouped 5-fold cross-validation (folds split by configuration, so the four replicates of a
configuration never straddle train/test), five surrogates + XGB/GPR blend.
Outputs: results/cv_metrics_per_fold.csv, cv_summary.csv, cv_hyperparameters.csv,
         cv_oof_predictions.csv, final_models.joblib (final XGB + GPR fits and blend weights)."""
import collections, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from common import *

s, tr, va = load()
X, y, g = tr[FEATURES].values, tr.Fmax_N.values, tr.configuration.values
MODELS = ["XGB", "RF", "GPR", "SVR", "Ridge"]
folds = list(GroupKFold(5).split(X, y, g))
print("fold sizes (test/train specimens, test configs):",
      [(len(te), len(tr_), len(set(g[te]))) for tr_, te in folds])


def tuned(name, Xa, ya, ga):
    s_ = RandomizedSearchCV(make_model(name), PARAM_GRID[name], n_iter=12, cv=GroupKFold(3),
                            scoring="r2", random_state=SEED)
    s_.fit(Xa, ya, groups=ga)
    return s_


oof = {n: np.zeros(len(y)) for n in MODELS}
hp = []
for k, (a, b) in enumerate(folds, 1):
    for n in MODELS:
        if n in PARAM_GRID:
            sr = tuned(n, X[a], y[a], g[a]); m = sr.best_estimator_
            hp.append(dict(fold=k, model=n, params=str(sr.best_params_)))
        else:
            m = make_model(n).fit(X[a], y[a])
            hp.append(dict(fold=k, model=n, params=str(m.named_steps["gpr"].kernel_)))
        oof[n][b] = m.predict(X[b])
    print("fold", k, "done", flush=True)

# blend weights: inverse mean fold-RMSE of XGB and GPR (as in the paper)
def fold_metrics(p):
    return np.array([[r2_score(y[b], p[b]), mean_squared_error(y[b], p[b]) ** .5,
                      mean_absolute_error(y[b], p[b])] for _, b in folds])
met = {n: fold_metrics(oof[n]) for n in MODELS}
w = 1 / np.array([met["XGB"][:, 1].mean(), met["GPR"][:, 1].mean()]); w /= w.sum()
oof["Blend"] = w[0] * oof["XGB"] + w[1] * oof["GPR"]
met["Blend"] = fold_metrics(oof["Blend"])
print("blend weights XGB/GPR: %.2f / %.2f" % tuple(w))

rows, summ = [], []
for n, a in met.items():
    for k in range(5):
        rows.append(dict(model=n, fold=k + 1, R2=a[k, 0], RMSE_N=a[k, 1], MAE_N=a[k, 2]))
    summ.append(dict(model=n, R2_mean=a[:, 0].mean(), R2_SD=a[:, 0].std(ddof=1),
                     RMSE_mean_N=a[:, 1].mean(), RMSE_SD_N=a[:, 1].std(ddof=1),
                     MAE_mean_N=a[:, 2].mean(), MAE_SD_N=a[:, 2].std(ddof=1)))
S = pd.DataFrame(summ)
print(S.round(3).to_string(index=False))
pd.DataFrame(rows).round(5).to_csv(RES / "cv_metrics_per_fold.csv", index=False)
S.round(5).to_csv(RES / "cv_summary.csv", index=False)
pd.DataFrame(hp).to_csv(RES / "cv_hyperparameters.csv", index=False)
fold_id = np.zeros(len(y), int)
for k, (_, b) in enumerate(folds, 1): fold_id[b] = k
O = pd.DataFrame(dict(specimen_id=tr.specimen_id, configuration=g, cv_fold=fold_id, Fmax_N=y,
                      **{f"oof_pred_{n}_N": oof[n] for n in MODELS + ["Blend"]}))
O.round(4).to_csv(RES / "cv_oof_predictions.csv", index=False)

# final fits on all 160 specimens
final = {n: tuned(n, X, y, g).best_estimator_ for n in ["XGB"]}
final["GPR"] = make_model("GPR").fit(X, y)
joblib.dump(dict(models=final, weights=w, features=FEATURES, F_plus=float(tr.groupby("configuration").Fmax_N.mean().max())),
            RES / "final_models.joblib")
print("final XGB params:", final["XGB"].get_params()["max_depth"], final["XGB"].get_params()["learning_rate"],
      final["XGB"].get_params()["n_estimators"], final["XGB"].get_params()["min_child_weight"])
print("final GPR kernel:", final["GPR"].named_steps["gpr"].kernel_)
