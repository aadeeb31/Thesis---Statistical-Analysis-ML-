"""Failure-mode statistics: chi-square, Monte Carlo permutation tests (specimens and configurations),
Wilson intervals, Fisher exact tests.  Output: results/failure_mode_stats.json, failure_mode_tables.csv"""
import json
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportion_confint
from common import load, RES

s, tr, va = load()
rng = np.random.default_rng(7)
fm, fam = "failure_mode_ASTM", "topology_type"
pd.crosstab(tr[fam], tr[fm]).to_csv(RES / "failure_mode_tables.csv")
obs = chi2_contingency(pd.crosstab(tr[fam], tr[fm]).values, correction=False)[0]
codes, fcodes = pd.factorize(tr[fm])[0], pd.factorize(tr[fam])[0]


def chi_codes(f, m):
    c = np.zeros((f.max() + 1, m.max() + 1)); np.add.at(c, (f, m), 1)
    e = np.outer(c.sum(1), c.sum(0)) / c.sum(); return ((c - e) ** 2 / e).sum()


N = 100000
o1 = chi_codes(fcodes, codes)
ge1 = sum(chi_codes(fcodes, rng.permutation(codes)) >= o1 - 1e-9 for _ in range(N))
cf = tr.groupby("configuration").topology_type.first(); famc = pd.factorize(cf.values)[0]
ci = tr.configuration.map({c: i for i, c in enumerate(cf.index)}).values
ge2 = sum(chi_codes(famc[rng.permutation(len(cf))][ci], codes) >= o1 - 1e-9 for _ in range(N))
n = len(tr); V = np.sqrt(obs / (n * (min(pd.crosstab(tr[fam], tr[fm]).shape) - 1)))
res = dict(chi2=obs, cramers_V=V, p_mc_specimen=(ge1 + 1) / (N + 1), p_mc_configuration=(ge2 + 1) / (N + 1), N_perm=N)
cohes = {"CF", "CF-M", "CF-IL"}


def wil(k, n):
    lo, hi = proportion_confint(int(k), n, method="wilson"); return (int(k), n, lo, hi)


res["val_cohesive"] = wil(va[fm].isin(cohes).sum(), 16)
res["val_CF-IL"] = wil((va[fm] == "CF-IL").sum(), 16)
res["val_AF"] = wil((va[fm] == "AF").sum(), 16)
res["val_AF-M"] = wil((va[fm] == "AF-M").sum(), 16)
bd = va[va.configuration.isin(["V-β", "V-δ"])]; res["VbVd_CF-IL"] = wil((bd[fm] == "CF-IL").sum(), 8)
t45 = tr[tr[fam] == "auxetic_45deg"]
k1, k2 = (va[fm] == "CF-IL").sum(), (t45[fm] == "CF-IL").sum()
res["fisher_CF-IL_val_vs_45train"] = fisher_exact([[k1, 16 - k1], [k2, 36 - k2]])[1]; res["train45_CF-IL"] = (int(k2), 36)
k1, k2 = va[fm].isin(cohes).sum(), t45[fm].isin(cohes).sum()
res["fisher_cohesive_val_vs_45train_unadjusted"] = fisher_exact([[k1, 16 - k1], [k2, 36 - k2]])[1]; res["train45_cohesive"] = (int(k2), 36)
E_ = chi2_contingency(pd.crosstab(tr[fam], tr[fm]).values)[3]
res["share_expected_lt5"] = float((E_ < 5).mean()); res["n_expected_lt1"] = int((E_ < 1).sum())
json.dump(res, open(RES / "failure_mode_stats.json", "w"), default=float, indent=1)
print(json.dumps(res, default=float, indent=1))
