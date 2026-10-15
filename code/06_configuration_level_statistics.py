"""Configuration-level analyses that respect the four-replicate clustering of the 160 specimens:
Pearson/Spearman r with cluster bootstrap, OLS vs random-intercept mixed model, 45 deg vs Base,
two-way ANOVA on the 27 uniform auxetic configurations.
Outputs: results/config_level_stats.json, config_anova.json"""
import json
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.formula.api import ols
from common import load, RES

s, tr, va = load()
v = tr.rename(columns=dict(Fmax_N="F", configuration="cfg", interlayer_IL_mm="IL", strut_t_mm="strut", theta_deg="th"))
rci = lambda r, n: (np.tanh(np.arctanh(r) - 1.96 / np.sqrt(n - 3)), np.tanh(np.arctanh(r) + 1.96 / np.sqrt(n - 3)))
g = v.groupby("cfg").agg(AI=("AI", "first"), F=("F", "mean"), fam=("fam", "first"), IL=("IL", "first")).reset_index()
res = {}
r = np.corrcoef(v.AI, v.F)[0, 1]; res["r_spec160"] = (r, *rci(r, 160))
rc = np.corrcoef(g.AI, g.F)[0, 1]; res["r_cfg40"] = (rc, *rci(rc, 40), len(g))
sp = stats.spearmanr(g.AI, g.F); res["rho_cfg40"] = (sp[0], sp[1])
uni = ["Ax_45", "Ax_60", "Ax_75"]
u = g[g.fam.isin(uni)]; ru = np.corrcoef(u.AI, u.F)[0, 1]; res["r_cfg27"] = (ru, *rci(ru, len(u)), len(u))
vu = v[v.fam.isin(uni)]; res["r_spec108"] = (np.corrcoef(vu.AI, vu.F)[0, 1],)
m0 = smf.ols("F~AI", v).fit(); res["ols_AI_per0.1"] = (m0.params["AI"] * .1, *(m0.conf_int().loc["AI"] * .1))
mm = smf.mixedlm("F~AI", v, groups=v.cfg).fit(reml=True)
res["mixed_AI_per0.1"] = (mm.params["AI"] * .1, *(mm.conf_int().loc["AI"] * .1), float(mm.cov_re.iloc[0, 0] ** .5), float(mm.scale ** .5))
m1 = smf.ols("F~AI+IL", v).fit(); mm1 = smf.mixedlm("F~AI+IL", v, groups=v.cfg).fit(reml=True)
res["ols_IL_N_per_mm"] = (m1.params["IL"], *m1.conf_int().loc["IL"]); res["mixed_IL_N_per_mm"] = (mm1.params["IL"], *mm1.conf_int().loc["IL"])
a = g[g.fam == "Ax_45"].F.values; b = v[v.fam == "Base"].F.values
t = stats.ttest_ind(a, b, equal_var=False); d = a.mean() - b.mean()
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
dfw = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
res["d45_base_config_means"] = (d, d / b.mean() * 100, d - stats.t.ppf(.975, dfw) * se, d + stats.t.ppf(.975, dfw) * se, t.pvalue, len(a), len(b))
w = v[v.fam.isin(["Ax_45", "Base"])].copy(); w["f45"] = (w.fam == "Ax_45").astype(int)
mw = smf.mixedlm("F~f45", w, groups=w.cfg).fit(reml=True)
res["mixed_45_minus_base"] = (mw.params["f45"], *mw.conf_int().loc["f45"], mw.pvalues["f45"])
rng = np.random.default_rng(1); bs = []
for _ in range(10000):
    smp = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
    if smp.AI.nunique() > 1: bs.append(np.corrcoef(smp.AI, smp.F)[0, 1])
res["r_cfg_cluster_bootstrap_95CI"] = tuple(np.percentile(bs, [2.5, 97.5]))
json.dump(res, open(RES / "config_level_stats.json", "w"), default=float, indent=1)
for k, x in res.items(): print(k, [round(float(y), 4) for y in x])

u27 = v[v.fam.isin(uni)].copy()
g27 = u27.groupby("cfg").agg(F=("F", "mean"), th=("th", "first"), IL=("IL", "first"), t=("strut", "first")).reset_index()
a2 = sm.stats.anova_lm(ols("F~C(th)+C(IL)+C(th):C(IL)", g27).fit(), typ=2); print(a2)
res2 = {"cfg_anova": a2.to_dict()}
for th in (45, 60, 75):
    sub = g27[g27.th == th]; f, p = stats.f_oneway(*[sub[sub.IL == i].F.values for i in sorted(sub.IL.unique())])
    res2[f"IL_effect_within_{th}deg"] = (f, p); print(th, round(f, 2), round(p, 3))
m3 = smf.mixedlm("F~AI+IL", u27, groups=u27.cfg).fit(reml=True); o3 = smf.ols("F~AI+IL", u27).fit()
res2["mixed27"] = dict(AI_per_0p1=(m3.params["AI"] * .1, *(m3.conf_int().loc["AI"] * .1)),
    IL=(m3.params["IL"], *m3.conf_int().loc["IL"], m3.pvalues["IL"]))
res2["ols27"] = dict(AI_per_0p1=(o3.params["AI"] * .1, *(o3.conf_int().loc["AI"] * .1)), IL=(o3.params["IL"], *o3.conf_int().loc["IL"], o3.pvalues["IL"]))
json.dump(res2, open(RES / "config_anova.json", "w"), default=float, indent=1)
print(res2["mixed27"], res2["ols27"])
