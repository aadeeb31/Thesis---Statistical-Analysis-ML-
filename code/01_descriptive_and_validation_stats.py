"""Family-level statistics vs Base, one-way tests, validation-vs-training tests.
Writes results/descriptive_stats.csv, results/validation_tests.csv and prints the headline numbers."""
import numpy as np, pandas as pd
from scipy import stats
from common import load, RES, FAMILY_ORDER

s, tr, va = load()
F = "Fmax_N"
base = tr[tr.fam == "Base"][F].values


def hedges_d(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
    return d * J, (d - 1.96 * se) * J, (d + 1.96 * se) * J


rows = []
eb = tr[tr.fam == "Base"].E_tri_Nmm
for f in FAMILY_ORDER:
    g = tr[tr.fam == f]
    x = g[F].values
    r = dict(family=f, n=len(x), Fmax_mean_N=x.mean(), Fmax_SD_N=x.std(ddof=1),
             pct_vs_base=(x.mean() / base.mean() - 1) * 100,
             delta_mean_mm=g.delta_peak_mm.mean(), Etri_mean_Nmm=g.E_tri_Nmm.mean(),
             Etri_pct_vs_base=(g.E_tri_Nmm.mean() / eb.mean() - 1) * 100)
    if f != "Base":
        r["welch_p_vs_base"] = stats.ttest_ind(x, base, equal_var=False).pvalue
        r["hedges_g"], r["g_ci_lo"], r["g_ci_hi"] = hedges_d(x, base)
    rows.append(r)
T = pd.DataFrame(rows)
T["welch_p_holm"] = np.nan
m = T.welch_p_vs_base.notna()
p = T.loc[m, "welch_p_vs_base"].values
order = np.argsort(p)
adj = np.empty_like(p)
run = 0
for rank, i in enumerate(order):
    run = max(run, (len(p) - rank) * p[i])
    adj[i] = min(1, run)
T.loc[m, "welch_p_holm"] = adj
T.round(4).to_csv(RES / "descriptive_stats.csv", index=False)
print(T.round(3).to_string())

grp = [tr[tr.fam == f][F].values for f in FAMILY_ORDER]
an = stats.f_oneway(*grp)
ss_b = sum(len(g) * (g.mean() - tr[F].mean()) ** 2 for g in grp)
print("\nOne-way ANOVA F=%.2f p=%.2e eta2=%.3f | Kruskal-Wallis H=%.1f p=%.2e | Levene p=%.3f" % (
    an.statistic, an.pvalue, ss_b / ((tr[F] - tr[F].mean()) ** 2).sum(),
    *stats.kruskal(*grp), stats.levene(*grp).pvalue))
th = stats.tukey_hsd(*grp)
print("Tukey HSD pairs with p<0.05:",
      [(FAMILY_ORDER[i], FAMILY_ORDER[j], round(th.pvalue[i, j], 4))
       for i in range(8) for j in range(i + 1, 8) if th.pvalue[i, j] < 0.05])

cm = tr.groupby("configuration").agg(AI=("AI", "first"), F=(F, "mean"), fam=("fam", "first"))
best = cm.F.idxmax()
print("\nBest training configuration:", best, round(cm.F.max(), 1), "N (+%.1f%% vs Base)" % ((cm.F.max() / base.mean() - 1) * 100))
print("Pearson r(AI, Fmax): specimen level %.3f, configuration level %.3f" % (
    np.corrcoef(tr.AI, tr[F])[0, 1], np.corrcoef(cm.AI, cm.F)[0, 1]))

# ---- validation
bestF = tr[tr.configuration == best][F].values
fam45 = tr[tr.fam == "Ax_45"][F]
rows = []
for c, g in va.groupby("configuration", sort=False):
    x = g[F]
    rows.append(dict(candidate=c, n=len(x), Fmax_mean_N=x.mean(), Fmax_SD_N=x.std(ddof=1),
                     pct_vs_base=(x.mean() / base.mean() - 1) * 100,
                     welch_p_vs_base=stats.ttest_ind(x, base, equal_var=False).pvalue,
                     pct_vs_best_training=(x.mean() / bestF.mean() - 1) * 100,
                     welch_p_vs_best_training=stats.ttest_ind(x, bestF, equal_var=False).pvalue,
                     pct_vs_45family=(x.mean() / fam45.mean() - 1) * 100,
                     welch_p_vs_45family=stats.ttest_ind(x, fam45, equal_var=False).pvalue,
                     Etri_mean_Nmm=g.E_tri_Nmm.mean(),
                     Etri_pct_vs_base=(g.E_tri_Nmm.mean() / eb.mean() - 1) * 100))
V = pd.DataFrame(rows)
V.round(4).to_csv(RES / "validation_tests.csv", index=False)
print("\n", V.round(3).to_string())
