"""Additional benchmark regressors (post-hoc, exploratory; Supplementary Table S16).
Same data, inputs, grouped 5-fold split (by configuration) and tuning protocol as 02_cross_validation.py:
RandomizedSearchCV, 12 draws, inner GroupKFold(3), scoring R2. The target is standardised inside each
training fold. Mean |SHAP| (N) is computed on a final fit to all 160 specimens (TreeExplainer for the tree
models, exact Shapley with the full data as background for the others).
Needs: lightgbm, catboost (see requirements.txt).
Outputs: results/extra_models_cv_summary.csv, results/extra_models_shap.csv"""
import numpy as np, pandas as pd, shap
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from common import *

s, tr, va = load()
X, y, g = tr[FEATURES].values, tr.Fmax_N.values, tr.configuration.values
folds = list(GroupKFold(5).split(X, y, g))
sc = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])
# name: (estimator, search space, closest manuscript model, SHAP method)
M = {
 "CatBoost": (CatBoostRegressor(random_seed=SEED, verbose=0, thread_count=4, allow_writing_files=False),
              dict(iterations=[300, 600], depth=[3, 4, 6], learning_rate=[.03, .05, .1]), "XGBoost", "tree"),
 "LightGBM": (lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=4, min_child_samples=5),
              dict(n_estimators=[200, 400], learning_rate=[.03, .05, .1], num_leaves=[4, 8, 16], min_child_samples=[3, 5, 10]), "XGBoost", "tree"),
 "GradientBoosting": (GradientBoostingRegressor(random_state=SEED),
              dict(n_estimators=[100, 200, 400], learning_rate=[.03, .05, .1], max_depth=[2, 3, 4], subsample=[.7, 1.0]), "XGBoost", "tree"),
 "AdaBoost": (AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), random_state=SEED),
              dict(n_estimators=[50, 100, 200], learning_rate=[.05, .1, .5, 1.0]), "XGBoost", "exact"),
 "ExtraTrees": (ExtraTreesRegressor(random_state=SEED, n_jobs=4),
              dict(n_estimators=[200, 400], max_depth=[None, 8, 16], max_features=["sqrt", 1.0]), "Random Forest", "tree"),
 "DecisionTree": (DecisionTreeRegressor(random_state=SEED),
              dict(max_depth=[3, 5, 8, None], min_samples_leaf=[1, 3, 5, 10]), "Random Forest", "tree"),
 "KNN": (sc(KNeighborsRegressor()), dict(m__n_neighbors=[3, 5, 8, 12, 20], m__weights=["uniform", "distance"]), "GPR", "exact"),
 "KernelRidge-RBF": (sc(KernelRidge(kernel="rbf")), dict(m__alpha=[.01, .1, 1], m__gamma=[.01, .05, .1, .3]), "SVR", "exact"),
 "MLP": (sc(MLPRegressor(max_iter=3000, random_state=SEED)),
              dict(m__hidden_layer_sizes=[(16,), (32,), (32, 16)], m__alpha=[1e-4, 1e-2, 1], m__learning_rate_init=[.003, .01]), "none", "exact"),
 "ElasticNet": (sc(ElasticNet(max_iter=20000)), dict(m__alpha=[.01, .1, 1, 10], m__l1_ratio=[.1, .5, .9]), "Ridge", "exact"),
}

def tuned(name, Xa, ya, ga):
    est, grid = M[name][:2]
    r = RandomizedSearchCV(est, grid, n_iter=12, cv=GroupKFold(3), scoring="r2", random_state=SEED)
    return r.fit(Xa, ya, groups=ga)

rows, shp = [], []
for n in M:
    oof = np.zeros(len(y))
    for a, b in folds:
        mu, sd = y[a].mean(), y[a].std()
        oof[b] = tuned(n, X[a], (y[a] - mu) / sd, g[a]).predict(X[b]) * sd + mu
    fm = np.array([[r2_score(y[b], oof[b]), mean_squared_error(y[b], oof[b]) ** .5, mean_absolute_error(y[b], oof[b])] for _, b in folds])
    a10 = float((np.abs(oof / y - 1) <= .10).mean())
    rows.append(dict(model=n, closest_manuscript_model=M[n][2], R2_mean=fm[:, 0].mean(), R2_SD=fm[:, 0].std(ddof=1),
                     RMSE_mean_N=fm[:, 1].mean(), MAE_mean_N=fm[:, 2].mean(), MSE_mean_N2=(fm[:, 1] ** 2).mean(), a10=a10))
    mu, sd = y.mean(), y.std()
    fin = tuned(n, X, (y - mu) / sd, g).best_estimator_
    if M[n][3] == "tree":
        sv = np.array(shap.TreeExplainer(fin).shap_values(X)) * sd
    else:
        sv = shap.explainers.Exact(lambda Z: fin.predict(Z) * sd, shap.maskers.Independent(X, max_samples=len(X)))(X).values
    shp.append(dict(model=n, **{f"meanabsSHAP_{f}_N": v for f, v in zip(SHORT, np.abs(sv).mean(0))}))
    print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).sort_values("R2_mean", ascending=False).round(5).to_csv(RES / "extra_models_cv_summary.csv", index=False)
pd.DataFrame(shp).round(3).to_csv(RES / "extra_models_shap.csv", index=False)
