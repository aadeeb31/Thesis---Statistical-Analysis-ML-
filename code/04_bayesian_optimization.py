"""Expected-improvement sweep and retrospective re-prediction of the four validation candidates.
Surrogate = w*XGB + (1-w)*GPR (final fits from 02_cross_validation.py).
EI uses F+ = best configuration mean in the training data (2132.3 N) and xi = 1 N.
Outputs: results/ei_sweep.csv, results/retrospective_predictions.csv, results/ei_surface_top20.csv"""
import joblib
import numpy as np, pandas as pd
from scipy.stats import norm
from common import *

s, tr, va = load()
M = joblib.load(RES / "final_models.joblib")
xgbm, gprm = M["models"]["XGB"], M["models"]["GPR"]
w0, Fp = M["weights"][0], M["F_plus"]
gp, sc = gprm.named_steps["gpr"], gprm.named_steps["sc"]
print("F+ = %.1f N, blend weight XGB = %.3f" % (Fp, w0))


def predict(rows, w=w0):
    R = np.asarray(rows, float)
    mx = xgbm.predict(R); mg, sg = gp.predict(sc.transform(R), return_std=True)
    return w * mx + (1 - w) * mg, sg


def ei(m, sd, xi=1.0):
    z = (m - Fp - xi) / (sd + 1e-9)
    return (sd + 1e-9) * (z * norm.cdf(z) + norm.pdf(z))


# 41 x 41 x 11 grid over theta in [35,55] deg, IL in [0.6,1.4] mm, t in [0.4,0.6] mm
TH, IL, ST = np.linspace(35, 55, 41), np.linspace(0.6, 1.4, 41), np.linspace(0.4, 0.6, 11)
grid = [(a, b, c) for a in TH for b in IL for c in ST]
R = np.array([candidate_row(*p) for p in grid])
mx = xgbm.predict(R); mg, sg = gp.predict(sc.transform(R), return_std=True)
rows = []
for w in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    m = w * mx + (1 - w) * mg; e = ei(m, sg); i = int(np.argmax(e))
    rows.append(dict(w_XGB=w, theta_star_deg=grid[i][0], IL_star_mm=grid[i][1], t_star_mm=grid[i][2],
                     Fmax_star_N=m[i], EI_N=e[i]))
E_ = pd.DataFrame(rows); E_.round(3).to_csv(RES / "ei_sweep.csv", index=False)
print(E_.round(2).to_string(index=False))
m = w0 * mx + (1 - w0) * mg; e = ei(m, sg); top = np.argsort(-e)[:20]
pd.DataFrame([dict(theta_deg=grid[i][0], IL_mm=grid[i][1], t_mm=grid[i][2], pred_N=m[i], sigma_N=sg[i], EI_N=e[i]) for i in top])\
    .round(3).to_csv(RES / "ei_surface_top20.csv", index=False)

# retrospective predictions (final model, measured porosity)
CAND = {"V-α": (43, 1.1, 0.4), "V-β": (40, 0.8, 0.4), "V-δ": (38, 0.8, 0.6), "V-ε": (45, 0.8, 0.4)}
rows = []
for k, (th, il, st) in CAND.items():
    g = va[va.configuration == k]
    por = float(g.porosity_pct.iloc[0])
    pm, ps = predict([candidate_row(th, il, st, por)])
    pm_a, ps_a = predict([candidate_row(th, il, st)])
    rows.append(dict(candidate=k, theta_deg=th, IL_mm=il, t_mm=st, measured_mean_N=g.Fmax_N.mean(),
                     retro_pred_N=pm[0], retro_sigma_N=ps[0], retro_gap_pct=(g.Fmax_N.mean() / pm[0] - 1) * 100,
                     approx_porosity_pred_N=pm_a[0]))
Rr = pd.DataFrame(rows); Rr.round(3).to_csv(RES / "retrospective_predictions.csv", index=False)
print(Rr.round(2).to_string(index=False))
print("mean |retrospective gap| = %.1f %%" % Rr.retro_gap_pct.abs().mean())
