"""Compares regenerated results with the values reported in the manuscript. Exit code 1 if any differ."""
import json, sys
import pandas as pd
from common import RES

ok = True
def check(label, got, want, tol):
    global ok
    good = abs(got - want) <= tol; ok &= good
    print(("OK   " if good else "FAIL ") + f"{label}: {got:.4g} (reported {want})")

S = pd.read_csv(RES / "cv_summary.csv").set_index("model")
check("GPR grouped-CV R2", S.loc["GPR", "R2_mean"], 0.874, 0.0006)
check("GPR grouped-CV RMSE (N)", S.loc["GPR", "RMSE_mean_N"], 74.8, 0.06)
check("Blend grouped-CV R2", S.loc["Blend", "R2_mean"], 0.855, 0.0006)
I = pd.read_csv(RES / "shap_gpr_importance.csv").set_index("input")
check("GPR SHAP, AI (N)", I.loc["AI", "mean_abs_SHAP_N"], 149.8, 0.06)
R = pd.read_csv(RES / "retrospective_predictions.csv")
check("mean |retrospective gap| (%)", R.retro_gap_pct.abs().mean(), 6.6, 0.06)
V = pd.read_csv(RES / "validation_tests.csv").set_index("candidate")
check("V-delta mean Fmax (N)", V.loc["V-δ", "Fmax_mean_N"], 2230.6, 0.06)
D = pd.read_csv(RES / "descriptive_stats.csv").set_index("family")
check("45 deg family vs Base (%)", D.loc["Ax_45", "pct_vs_base"], 24.5, 0.06)
F = json.load(open(RES / "failure_mode_stats.json"))
check("chi-square (35 df)", F["chi2"], 221.3, 0.06)
C = json.load(open(RES / "config_level_stats.json"))
check("configuration-level r(AI, Fmax)", C["r_cfg40"][0], 0.91, 0.006)
check("45 deg minus Base, mixed model (N)", C["mixed_45_minus_base"][0], 402, 0.6)
print("\nAll reported values reproduced." if ok else "\nMISMATCH - see FAIL lines.")
sys.exit(0 if ok else 1)
