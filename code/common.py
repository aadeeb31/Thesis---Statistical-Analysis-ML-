"""Shared helpers: data loading, model definitions, hyper-parameter grids.
All scripts read data/specimens.csv only and write to results/."""
import math, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "specimens.csv"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 42
# model inputs, in the order used throughout the paper
FEATURES = ["AI", "porosity_pct", "interlayer_IL_mm", "epoxy_thickness_mm",
            "strut_t_mm", "GR", "CAR"]
SHORT = ["AI", "por", "IL", "epoxy", "t", "GR", "CAR"]
FAMILY_ORDER = ["Base", "Ax_45", "Ax_60", "Ax_75", "FGax_T1", "FGax_T2", "Flat", "Spacer"]


def load():
    """Return (all specimens, 160 training specimens, 16 validation specimens)."""
    s = pd.read_csv(DATA)
    s["fam"] = s.configuration.str.extract(r"^(Ax_\d+|FGax_T\d|Flat|Spacer|Base)")[0]
    tr = s[s.subset == "training"].reset_index(drop=True)
    va = s[s.subset == "validation"].reset_index(drop=True)
    return s, tr, va


# hyper-parameter search spaces (RandomizedSearchCV, 12 draws, inner GroupKFold(3))
PARAM_GRID = {
    "XGB": dict(max_depth=[3, 5, 7], learning_rate=[0.05, 0.1, 0.2],
                n_estimators=[200, 400, 800], min_child_weight=[1, 3, 5]),
    "RF": dict(n_estimators=[200, 400], max_depth=[None, 8, 16], max_features=["sqrt", 1.0]),
    "SVR": dict(svr__C=[0.1, 1, 10, 100, 1000], svr__gamma=["scale", 0.01, 0.1]),
    "Ridge": dict(ridge__alpha=[0.01, 0.1, 1, 10, 100]),
}


def make_gpr(n_features=len(FEATURES)):
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=2.5)
              + WhiteKernel(1e-2))
    return Pipeline([("sc", StandardScaler()),
                     ("gpr", GaussianProcessRegressor(kernel, n_restarts_optimizer=3,
                                                      normalize_y=True, random_state=SEED))])


def make_model(name):
    if name == "XGB":
        return xgb.XGBRegressor(random_state=SEED, verbosity=0, n_jobs=4)
    if name == "RF":
        return RandomForestRegressor(random_state=SEED, n_jobs=4)
    if name == "SVR":
        return Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", epsilon=0.03))])
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("ridge", Ridge())])
    if name == "GPR":
        return make_gpr()
    raise ValueError(name)


def ai_of_theta(theta_deg):
    """Auxetic index of the uniform re-entrant cell as a function of re-entrant angle."""
    t = math.radians(theta_deg)
    return math.cos(t) ** 2 / (math.sin(t) * (1 + math.sin(t)))


def porosity_heuristic(ai):
    """Selection-time porosity approximation used in the optimiser: clip(AI/0.414*82, 0, 95)."""
    return float(np.clip(ai / 0.414 * 82.0, 0, 95))


def candidate_row(theta, il, t, porosity=None):
    """Feature vector of a candidate (epoxy 0.30 mm, GR = CAR = 1)."""
    a = ai_of_theta(theta)
    return [a, porosity_heuristic(a) if porosity is None else porosity, il, 0.30, t, 1.0, 1.0]
