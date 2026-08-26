# ═══════════════════════════════════════════════════════════════════════════════
# SLJ_Analysis_Pro.py — v10.0 (reorganized & paper-aligned edition)
# Auxetic Re-Entrant Honeycomb PETG Single-Lap Joints
#
# Companion code for:
#   Chowdhury, M.A., Chowdhury, S., Islam, M.A., Kader, M.G. "Auxetic-Clamping
#   Suppression of Peel-Stress Failure in FDM-PETG Single-Lap Adhesive Joints:
#   A Closed-Loop, Machine-Learning-Guided Design Study."
#
#
# COLAB UPLOAD:  SLJ_ML_Database_v7.8.xlsx   (only this file is needed)
#
# OUTPUTS:
#   SLJ_Analysis_Figures.pdf    — Figs 1-7   (statistical analysis)
#   SLJ_Threshold_Figure.pdf    — Fig 5-style AI-threshold plot + bootstrap
#   SLJ_Supplementary.pdf       — Tukey HSD significance matrix (supp. only)
#   SLJ_ML_Figures.pdf          — Figs 8-15  (ML + SHAP + BO)
#   SLJ_DesignSpace_Figures.pdf — Figs 16-18 (design space extrapolation)
#   SLJ_Validation_Figures.pdf  — Physical validation (Figs 11-12 style)
#   SLJ_Stats_Tables.xlsx       — ANOVA, Levene/Shapiro/Kruskal, two-way ANOVA,
#                                 Tukey, Cohen's d, chi-square, bootstrap CI
#   SLJ_ML_Results.xlsx         — CV scores, tuned hyperparameters, SHAP,
#                                 BO tables, validation comparison
# ═══════════════════════════════════════════════════════════════════════════════

# ── PART A: SETUP ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import studentized_range, chi2_contingency, shapiro, levene, kruskal, f as f_dist
from scipy.optimize import brentq
from itertools import combinations
from collections import Counter
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import warnings, os
warnings.filterwarnings('ignore')
np.random.seed(42)

try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    COLAB = True
except ImportError:
    COLAB = False

# ── File paths ─────────────────────────────────────────────────────────────────
INPUT_FILE     = 'SLJ_DB1_Fixed_Formatted_v4_1.xlsx'
ANALYSIS_PDF   = 'SLJ_Analysis_Figures.pdf'
THRESHOLD_PDF  = 'SLJ_Threshold_Figure.pdf'
SUPP_PDF       = 'SLJ_Supplementary.pdf'
ML_PDF         = 'SLJ_ML_Figures.pdf'
DESIGN_PDF     = 'SLJ_DesignSpace_Figures.pdf'
VALIDATION_PDF = 'SLJ_Validation_Figures.pdf'
STATS_XLS      = 'SLJ_Stats_Tables.xlsx'
ML_XLS         = 'SLJ_ML_Results.xlsx'

# ── Runtime knobs ───────────────────────────────────────────────────────────────
# Full run (paper-matching) takes a while, mostly in the hyperparameter search
# and the 10,000-resample bootstrap. Override any of these from the shell for
# a fast smoke test, e.g.: SLJ_N_BOOTSTRAP=200 SLJ_N_ITER_SEARCH=3 python SLJ_Analysis_Pro.py
N_BOOTSTRAP     = int(os.environ.get('SLJ_N_BOOTSTRAP', 10000))    # Sec. 2.6/3.1: paper uses 10,000
N_ITER_SEARCH   = int(os.environ.get('SLJ_N_ITER_SEARCH', 15))     # RandomizedSearchCV draws/fold
INNER_CV_FOLDS  = int(os.environ.get('SLJ_INNER_CV_FOLDS', 3))     # "3-fold randomized search", Sec. 4.1

# ── Font detection ─────────────────────────────────────────────────────────────
def _detect_font():
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for candidate in ['Times New Roman', 'Times', 'DejaVu Serif']:
        if candidate in names:
            return candidate
    return 'DejaVu Serif'
TNR = _detect_font()

# ── Elsevier / Composites Part B rcParams ─────────────────────────────────────
#   Rules: Times New Roman, no bold axis labels, inward ticks, 0.9 linewidth,
#          300 DPI, no embedded titles in final figures
plt.rcParams.update({
    'font.family':          'serif',
    'font.serif':           [TNR, 'Times', 'DejaVu Serif'],
    'font.size':            9,
    'axes.labelsize':       10.5,
    'axes.labelweight':     'normal',   # NOT bold — Elsevier standard
    'axes.titlesize':       10,
    'axes.titleweight':     'normal',
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'legend.fontsize':      8.5,
    'legend.framealpha':    1.0,
    'legend.edgecolor':     'black',
    'figure.dpi':           300,
    'figure.facecolor':     'white',
    'axes.facecolor':       'white',
    'axes.grid':            False,
    'axes.linewidth':       0.9,
    'xtick.major.width':    0.9,
    'ytick.major.width':    0.9,
    'xtick.major.size':     4,
    'ytick.major.size':     4,
    'xtick.direction':      'in',       # inward ticks — Elsevier standard
    'ytick.direction':      'in',
    'lines.linewidth':      1.8,
    'patch.edgecolor':      'black',
    'patch.linewidth':      0.9,
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
})

# ── Okabe-Ito colour-blind-safe palette ───────────────────────────────────────
GPAL = {
    'Ax_45':   '#0072B2',   # Blue
    'Ax_60':   '#009E73',   # Bluish-green
    'Ax_75':   '#D55E00',   # Vermillion
    'FGax_T1': '#56B4E9',   # Sky-blue
    'FGax_T2': '#CC79A7',   # Reddish-purple
    'Base':    '#000000',   # Black
    'Flat':    '#999999',   # Grey
    'Spacer':  '#333333',   # Dark grey
}
GLABELS = {
    'Ax_45': '45°', 'Ax_60': '60°', 'Ax_75': '75°',
    'FGax_T1': 'FGax-T1', 'FGax_T2': 'FGax-T2',
    'Base': 'Base', 'Flat': 'Flat', 'Spacer': 'Spacer',
}
GROUP_ORDER = ['Ax_45','Ax_60','Ax_75','FGax_T1','FGax_T2','Base','Flat','Spacer']

MODEL_COLORS = {
    'XGBoost':     '#0072B2',
    'RandomForest':'#009E73',
    'GPR':         '#D55E00',
    'SVR':         '#CC79A7',
    'Ridge':       '#56B4E9',
}
FEAT_COLORS = ['#0072B2','#009E73','#D55E00','#CC79A7','#56B4E9','#999999','#333333']
IL_COLORS_DS = {1.0: '#D55E00', 0.8: '#E69F00', 0.6: '#009E73'}
VAL_COLORS = {'V_alpha': '#7B3F00', 'V_beta': '#B8860B', 'V_delta': '#DC143C', 'V_epsilon': '#4B0082'}
VAL_LABELS = {'V_alpha': 'V-α', 'V_beta': 'V-β', 'V_delta': 'V-δ', 'V_epsilon': 'V-ε'}

# ── Polish function (no bold on axis labels — Elsevier standard) ──────────────
def polish(ax, xlabel=None, ylabel=None, title=None, grid=False):
    """Apply Elsevier-standard spine/tick formatting."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('black')
        ax.spines[sp].set_linewidth(0.9)
    ax.tick_params(axis='both', colors='black', width=0.9, length=4,
                   direction='in')
    if grid:
        ax.yaxis.grid(True, linewidth=0.4, color='#CCCCCC', linestyle='--',
                      zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, labelpad=4, fontfamily=TNR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, rotation=90, labelpad=5,
                      fontfamily=TNR)
    if title:
        ax.set_title(title, fontsize=10, pad=5, fontfamily=TNR)

# ── Re-entrant honeycomb geometry (Masters-Evans, Eqs. 1-2, Sec. 2.2) ─────────
# Defined here (Part A) rather than only in Part G, because the fixed BO
# section in Part D now needs it too, to parameterize candidates by theta
# (matching Sec. 4.2) instead of gridding AI directly.
def ai_of_theta(theta_deg):
    """Masters-Evans auxeticity index AI(theta) at h = l (Eq. 1-2)."""
    t = math.radians(theta_deg)
    return math.cos(t)**2 / (math.sin(t)*(1.0+math.sin(t)))

def theta_of_ai(ai_val, lo=1.0, hi=88.9):
    """Inverse of ai_of_theta by root-finding (monotonic decreasing on (0,90))."""
    try:
        return brentq(lambda t: ai_of_theta(t)-ai_val, lo, hi)
    except Exception:
        return np.nan

def ai_to_porosity_pct(ai_val, ai_ref=0.414, por_ref=82.0, cap=95.0):
    """Rough AI -> porosity mapping calibrated against the 45 deg training
    cell (AI=0.414 -> ~82% porosity, Table 1). Used only to keep synthetic BO
    candidates dimensionally consistent with the trained feature vector; the
    physical porosity in the real dataset is a direct CAD/measurement column."""
    return float(np.clip((ai_val/ai_ref)*por_ref, 0, cap))

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel(INPUT_FILE, sheet_name='ML_Dataset', header=1)
print(f"Loaded: {len(df)} specimens x {df.shape[1]} columns")

# DB1 topology_type -> v7.8 group alias (keeps all downstream GROUP_* dicts working)
_topo_map = {'auxetic_45deg':'Ax_45','auxetic_60deg':'Ax_60','auxetic_75deg':'Ax_75',
             'fgax_t1':'FGax_T1','fgax_t2':'FGax_T2','base':'Base','flat':'Flat',
             'spacer_2mm':'Spacer'}
df['group'] = df['topology_type'].map(_topo_map)

# ── Split off the physical-validation cohort (Sec. 5) ─────────────────────────
# The four out-of-grid Bayesian-Optimization candidates (V-alpha/beta/delta/
# epsilon, Table 5) live in the same 'ML_Dataset' sheet as the 160 training
# specimens (df['group'] is NaN for them, since they don't belong to any of
# the 8 training topology families). Everything in Parts B-H below operates
# on df_train only, exactly reproducing the 160-specimen training campaign of
# Sections 3-4; Part I brings df_val back in for the Section 5 comparison.
#
# NOTE: match the four candidate names against whatever your 'configuration'
# column actually spells them as. This regex accepts 'V_alpha'/'V-alpha'/'Va'
# style variants case-insensitively; adjust VAL_NAME_MAP below if your sheet
# uses something else (e.g. literal Greek letters 'V-α').
_val_mask = df['configuration'].astype(str).str.contains(
    r'^V[-_]?(alpha|beta|delta|epsilon|α|β|δ|ε)', case=False, regex=True, na=False)
df_val   = df[_val_mask].copy()
df_train = df[~_val_mask].copy()
_greek2ascii = {'α':'alpha','β':'beta','δ':'delta','ε':'epsilon'}
def _canon_val_name(s):
    s = str(s)
    for gk, gv in _greek2ascii.items():
        s = s.replace(gk, gv)
    s = s.replace('-', '_')
    m = {'alpha':'V_alpha','beta':'V_beta','delta':'V_delta','epsilon':'V_epsilon'}
    for k, v in m.items():
        if k in s.lower():
            return v
    return s
if len(df_val):
    df_val['val_id'] = df_val['configuration'].apply(_canon_val_name)
print(f"  -> {len(df_train)} training specimens, {len(df_val)} validation specimens "
      f"({sorted(df_val['val_id'].unique()) if len(df_val) else 'none found'})")

# From here on, 'df' refers to the training campaign only — matching every
# v9.0 computation, which was implicitly written against the 160-specimen set.
df = df_train

MECH = {
    'max_force_N':     'Peak Force (N)',
    'energy_Nmm':      'Fracture Energy (N·mm)',
    'displacement_mm': 'Displacement at Failure (mm)',
    'stiffness_N_mm':  'Stiffness (N/mm)',
}
# Sec. 3.1/Table 3 reports Fmax, Etri, ED, delta_peak, tau_max. This script's
# original MECH dict analyzes displacement_mm and stiffness_N_mm instead of
# ED and tau_max — kept as-is (extra, not a paper-reported ANOVA) since those
# two derived quantities (ED, tau_max) are simple deterministic rescalings of
# Fmax/Etri already visible in the Fmax/Etri results, not independent findings.

# ── Representative F-D curve generator (6-zone piecewise model) ───────────────
# NOTE: this is a *stylized* visualization aid (Fig. 1), not a measured curve —
# the paper does not describe this synthesis and Fig. 1 there is a schematic/
# geometry figure. Kept as an extra descriptive plot; skip if you'd rather show
# real per-specimen DAQ traces once available.
DD = (2/60)*0.01
GROUP_ALPHA = {'Ax_45':0.440,'Ax_60':0.460,'Ax_75':0.428,'FGax_T1':0.492,
               'FGax_T2':0.448,'Base':0.413,'Flat':0.430,'Spacer':0.420}
N_CORR  = {'Ax_45':1.653,'Ax_60':1.204,'Ax_75':1.371,'FGax_T1':1.059,
           'FGax_T2':1.264,'Base':1.427,'Flat':1.672,'Spacer':1.745}
TAIL_TYPE = {'Ax_45':'mixed','Ax_60':'adhesive_mixed','Ax_75':'adhesive_mixed',
             'FGax_T1':'adhesive_mixed','FGax_T2':'adhesive_mixed',
             'Base':'adhesive','Flat':'cohesive','Spacer':'cohesive'}
GNOISE  = {'Ax_45':0.018,'Ax_60':0.010,'Ax_75':0.006,'FGax_T1':0.009,
           'FGax_T2':0.008,'Base':0.005,'Flat':0.007,'Spacer':0.007}

def _smooth(x, w=7):
    return np.convolve(x, np.ones(w)/w, mode='same')

def gen_curve(F_pk, d_pk, grp, seed=42):
    rng = np.random.RandomState(seed)
    sigma = GNOISE.get(grp, 0.008)*F_pk
    n   = N_CORR.get(grp, 1.3)
    d_r = np.arange(0, d_pk+DD*0.5, DD); d_r[-1] = d_pk
    Fs  = F_pk*(d_r/d_pk)**n; Fs[0] = 0.0
    Fr  = np.clip(Fs + _smooth(rng.normal(0, sigma, len(d_r)), 9), 0, None)
    Fr[-1] = F_pk
    tail = TAIL_TYPE.get(grp, 'adhesive')
    if tail == 'adhesive':
        nt = rng.randint(3,7); dt = np.linspace(d_pk+DD, d_pk+nt*DD, nt)
        Ft = np.clip(F_pk*np.exp(-280*(dt-d_pk))+rng.normal(0,sigma*0.3,nt), 0, None)
    elif tail == 'adhesive_mixed':
        de = 0.03*d_pk; dt = np.arange(d_pk+DD, d_pk+de+DD, DD)
        if not len(dt): dt = np.array([d_pk+DD])
        Ft = np.clip(F_pk*np.exp(-4.5/de*(dt-d_pk))+rng.normal(0,sigma*0.6,len(dt)), 0, None)
    elif tail == 'mixed':
        de = 0.10*d_pk; dt = np.arange(d_pk+DD, d_pk+de+DD, DD)
        if not len(dt): dt = np.array([d_pk+DD])
        sp = int(len(dt)*0.85); Fp = F_pk*np.linspace(1.0, 0.38, len(dt))
        if len(dt)-sp > 0:
            Fp[sp:] = Fp[sp]*np.exp(-14*np.linspace(0, 1, len(dt)-sp))
        Ft = np.clip(Fp+_smooth(rng.normal(0,sigma,len(dt)),7)*(Fp/F_pk), 0, None)
    else:  # cohesive
        de = 0.14*d_pk; dt = np.arange(d_pk+DD, d_pk+de+DD, DD)
        if not len(dt): dt = np.array([d_pk+DD])
        tv = np.clip((dt-d_pk)/de, 0, 1)
        Ft = np.clip(F_pk*(1-tv)**1.5+rng.normal(0,sigma*0.6,len(dt)), 0, None)
    return np.concatenate([d_r, dt]), np.concatenate([Fr, Ft])

rep_curves = {}
for grp in GROUP_ORDER:
    sub = df[df['group'] == grp]
    if not len(sub): continue
    closest = sub.iloc[(sub['max_force_N']-sub['max_force_N'].mean()).abs().argsort().iloc[0]]
    d_arr, F_arr = gen_curve(float(closest['max_force_N']),
                              float(closest['displacement_mm']),
                              grp, abs(hash(closest['specimen_id']))%100000)
    rep_curves[grp] = (d_arr, F_arr)
print("  Representative curves generated")


# ── PART B: STATISTICAL ANALYSIS ──────────────────────────────────────────────
# Story: does topology change joint strength, and can we trust the test we
# used to say so? Sec. 2.6 lays out a decision tree (normality -> variance
# homogeneity -> pick the right omnibus test) before ever quoting a p-value;
# the v9.0 script skipped straight to stats.f_oneway. This section restores
# that decision tree, then keeps the original Tukey HSD / CLD / Cohen's d /
# chi-square block that was already faithful to the paper.

def welch_anova(groups_data):
    """Welch's ANOVA (unequal-variance F-test), classic formula.
    Returns (F, df1, df2, p)."""
    k = len(groups_data)
    ni = np.array([len(g) for g in groups_data], dtype=float)
    mi = np.array([g.mean() for g in groups_data])
    vi = np.array([g.var(ddof=1) for g in groups_data])
    wi = ni / vi
    sw = wi.sum()
    grand = (wi*mi).sum() / sw
    num = (wi*(mi-grand)**2).sum() / (k-1)
    A = ((1 - wi/sw)**2 / (ni-1)).sum() * (2*(k-2)) / (k**2 - 1) if k > 2 else \
        ((1 - wi/sw)**2 / (ni-1)).sum()  # k=2 edge case, avoid div by (k^2-1)-3-related issues
    denom_term = ((1 - wi/sw)**2 / (ni-1)).sum()
    F = num / (1 + (2*(k-2)/(k**2-1))*denom_term)
    df1 = k - 1
    df2 = (k**2 - 1) / (3*denom_term) if denom_term > 0 else np.inf
    p = 1 - f_dist.cdf(F, df1, df2)
    return F, df1, df2, p

def assumption_checked_omnibus(gdata, alpha=0.05):
    """Sec. 2.6 decision tree for one factor with >=2 groups:
      1) Shapiro-Wilk normality per group (skipped / assumed-normal for any
         group with n < 8, where the test is underpowered and the paper
         states normality is assumed for the small reference families).
      2) Levene's test for homogeneity of variance across all groups.
      3) If normal & homoscedastic -> one-way ANOVA is the test of record.
         If Levene flags heteroscedasticity -> also run Kruskal-Wallis
         (non-parametric robustness check) and Welch's ANOVA (unequal-
         variance parametric fallback), exactly as Sec. 2.6/3.1 describe.
    Returns a dict of every statistic computed, so nothing is thrown away.
    """
    shapiro_p = [shapiro(g)[1] if len(g) >= 8 else np.nan for g in gdata]
    normal_ok = all((p > alpha) or np.isnan(p) for p in shapiro_p)
    lev_stat, lev_p = levene(*gdata)
    homosced_ok = lev_p > alpha
    F, p = stats.f_oneway(*gdata)
    out = dict(shapiro_p_per_group=[round(float(p), 4) if not np.isnan(p) else None
                                     for p in shapiro_p],
               normal_ok=normal_ok, levene_stat=round(float(lev_stat), 4),
               levene_p=float(lev_p), homoscedastic_ok=homosced_ok,
               anova_F=round(float(F), 3), anova_p=float(p),
               kruskal_H=None, kruskal_p=None,
               welch_F=None, welch_df1=None, welch_df2=None, welch_p=None,
               primary_test='one-way ANOVA')
    if not homosced_ok:
        H, kp = kruskal(*gdata)
        wF, wdf1, wdf2, wp = welch_anova(gdata)
        out.update(kruskal_H=round(float(H), 3), kruskal_p=float(kp),
                    welch_F=round(float(wF), 3), welch_df1=round(float(wdf1), 3),
                    welch_df2=round(float(wdf2), 3), welch_p=float(wp),
                    primary_test='Welch ANOVA (+ Kruskal-Wallis robustness check)')
    return out

def cohens_d(x, y):
    sp = np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
    return (x.mean()-y.mean())/sp if sp > 0 else 0.0

def _d_label(d):
    a = abs(d)
    if a >= 1.2: return 'very large'
    if a >= 0.8: return 'large'
    if a >= 0.5: return 'medium'
    if a >= 0.2: return 'small'
    return 'negligible'

def tukey_hsd(groups_data, names, alpha=0.05):
    k = len(groups_data); N = sum(len(g) for g in groups_data)
    SSW = sum(np.sum((g-g.mean())**2) for g in groups_data)
    dfw = N-k; MSW = SSW/dfw
    rows = []
    for i, j in combinations(range(k), 2):
        ni, nj = len(groups_data[i]), len(groups_data[j])
        diff = groups_data[i].mean()-groups_data[j].mean()
        se = np.sqrt(MSW/2*(1/ni+1/nj)); q = abs(diff)/se if se > 0 else 0
        p  = max(0.0, 1-studentized_range.cdf(q, k, dfw))
        qc = studentized_range.ppf(1-alpha, k, dfw); ci = qc*se/np.sqrt(2)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        rows.append(dict(A=names[i], B=names[j],
                         mean_A=round(groups_data[i].mean(),2),
                         mean_B=round(groups_data[j].mean(),2),
                         diff=round(diff,2), CI_lo=round(diff-ci,2),
                         CI_hi=round(diff+ci,2), q=round(q,4),
                         p=float(p), sig=sig, significant=(p<alpha)))
    return pd.DataFrame(rows), MSW, dfw

def cld_from_tukey(tukey_df, group_order, col):
    ns_pairs = {frozenset([r.A, r.B]) for _, r in tukey_df.iterrows()
                if not r.significant}
    means    = {g: df[df['group']==g][col].mean() for g in group_order}
    sorted_g = sorted(group_order, key=lambda g: -means.get(g, 0))
    used = []; letters = {}; li = 0
    lc = lambda i: chr(97+i) if i < 26 else chr(96+i//26)+chr(97+i%26)
    for g in sorted_g:
        assigned = False
        for let, members in used:
            if all(frozenset([g, m]) in ns_pairs for m in members):
                letters[g] = letters.get(g, '')+let
                members.append(g); assigned = True; break
        if not assigned:
            let = lc(li); li += 1; letters[g] = let; used.append((let, [g]))
    return {g: letters.get(g, '') for g in group_order}

anova_rows = []; assumption_rows = []; tukey_all = {}; cohen_rows = []
for col, label in MECH.items():
    gdata = [df[df['group']==g][col].values for g in GROUP_ORDER]
    chk = assumption_checked_omnibus(gdata)
    assumption_rows.append(dict(response=label, col=col, **chk))
    F, p  = chk['anova_F'], chk['anova_p']
    gm    = np.concatenate(gdata).mean()
    SS_b  = sum(len(g)*(g.mean()-gm)**2 for g in gdata)
    SS_t  = sum(np.sum((g-gm)**2) for g in gdata)
    eta2  = SS_b/SS_t
    sig   = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    anova_rows.append(dict(response=label, col=col, F=round(F,3), p=float(p),
                           eta2=round(eta2,4), sig=sig,
                           df_bet=len(GROUP_ORDER)-1,
                           df_wit=len(df)-len(GROUP_ORDER),
                           primary_test=chk['primary_test'],
                           kruskal_H=chk['kruskal_H'], kruskal_p=chk['kruskal_p'],
                           welch_F=chk['welch_F'], welch_p=chk['welch_p']))
    tdf, _, _ = tukey_hsd(gdata, GROUP_ORDER)
    tdf['response'] = label; tdf['col'] = col; tukey_all[col] = tdf
    base_v = df[df['group']=='Base'][col].values
    for g in GROUP_ORDER:
        if g == 'Base': continue
        gv = df[df['group']==g][col].values; d = cohens_d(gv, base_v)
        n1, n2 = len(gv), len(base_v)
        se_d = np.sqrt((n1+n2)/(n1*n2)+d**2/(2*(n1+n2-2)))
        cohen_rows.append(dict(group=g, response=label, col=col,
                               d=round(d,4), CI_lo=round(d-1.96*se_d,4),
                               CI_hi=round(d+1.96*se_d,4),
                               mag=_d_label(d),
                               pct=round((gv.mean()-base_v.mean())/base_v.mean()*100,2)))

df_anova  = pd.DataFrame(anova_rows)
df_assum  = pd.DataFrame(assumption_rows)
df_cohen  = pd.DataFrame(cohen_rows)
df_tukey  = pd.concat(tukey_all.values(), ignore_index=True)
cld_force  = cld_from_tukey(tukey_all['max_force_N'], GROUP_ORDER, 'max_force_N')
cld_energy = cld_from_tukey(tukey_all['energy_Nmm'],  GROUP_ORDER, 'energy_Nmm')

ct = pd.crosstab(df['group'], df['failure_mode']).reindex(GROUP_ORDER)
ct = ct.loc[:, (ct>0).any()]
chi2_s, p_chi, dof_chi, _ = chi2_contingency(ct)
cramv = np.sqrt(chi2_s/(ct.values.sum()*(min(ct.shape)-1)))

_force_row = df_anova[df_anova['col']=='max_force_N'].iloc[0]
print(f"  Stats complete — Fmax: {_force_row['primary_test']}, "
      f"F={_force_row['F']:.1f}"
      + (f", Kruskal H={_force_row['kruskal_H']:.1f}" if pd.notna(_force_row['kruskal_H']) else "")
      + f"  chi2={chi2_s:.1f}  Cramer's V={cramv:.3f}")


# ── Two-way ANOVA on (theta, IL), Sec. 3.2 ────────────────────────────────────
# Restricted to the 27-cell auxetic factorial (45/60/75 deg), collapsed over
# strut thickness, exactly as Sec. 3.2 describes: "Second-order geometric
# effects are tested by two-way ANOVA on (theta, IL) with the associated
# interaction term." Reported for both Fmax (quoted explicitly in the text,
# F(2,99)=55.7 for theta) and Etri (not separately quoted, but the same
# decomposition is informative and costs nothing extra to compute).
try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False
    print("  statsmodels not found -- skipping two-way ANOVA (pip install statsmodels)")

twoway_rows = []
oneway_IL_rows = []
if _STATSMODELS:
    aux = df[df['group'].isin(['Ax_45','Ax_60','Ax_75'])].copy()
    aux['theta_cat'] = aux['group'].map({'Ax_45':'45','Ax_60':'60','Ax_75':'75'})
    aux['IL_cat']    = aux['interlayer_mm'].round(1).astype(str)
    for col, label in [('max_force_N','Peak Force (N)'), ('energy_Nmm','Fracture Energy (N·mm)')]:
        model = ols(f'{col} ~ C(theta_cat) * C(IL_cat)', data=aux).fit()
        tbl = anova_lm(model, typ=2)
        for term in ['C(theta_cat)', 'C(IL_cat)', 'C(theta_cat):C(IL_cat)']:
            if term in tbl.index:
                twoway_rows.append(dict(
                    response=label, col=col, term=term,
                    F=round(float(tbl.loc[term,'F']),3),
                    p=float(tbl.loc[term,'PR(>F)']),
                    df_num=round(float(tbl.loc[term,'df']),1),
                    df_resid=round(float(tbl.loc['Residual','df']),1)))
    # Per-angle one-way ANOVA across IL (quoted individually in Sec. 3.2:
    # F(2,33)=1.01 p=0.38 at 45 deg, etc.)
    for ang in [45, 60, 75]:
        sub = df[df['group']==f'Ax_{ang}']
        gdata_il = [sub[sub['interlayer_mm'].round(1)==il]['max_force_N'].values
                    for il in [0.6, 0.8, 1.0]]
        gdata_il = [g for g in gdata_il if len(g)]
        if len(gdata_il) >= 2:
            Fi, pi = stats.f_oneway(*gdata_il)
            oneway_IL_rows.append(dict(angle=ang, F=round(float(Fi),3), p=float(pi),
                                        df_bet=len(gdata_il)-1,
                                        df_wit=sum(len(g) for g in gdata_il)-len(gdata_il)))
df_twoway = pd.DataFrame(twoway_rows)
df_oneway_il = pd.DataFrame(oneway_IL_rows)
if len(df_twoway):
    print("  Two-way ANOVA (theta x IL) on Fmax:")
    print(df_twoway[df_twoway['col']=='max_force_N'].to_string(index=False))


# ── PART C: ANALYSIS FIGURES (Figs 1-7) ───────────────────────────────────────
# Story: eight bar charts and a heatmap, walking from "does the joint get
# stronger" (Figs 2-3) to "why" (Figs 4, 7) to "how does it break" (Fig 6).
print("\nGenerating analysis figures (1-7)...")

with PdfPages(ANALYSIS_PDF) as pdf:

    # ── Fig 1: Representative Force-Displacement Curves (extra/illustrative) ─
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for grp in GROUP_ORDER:
        if grp not in rep_curves: continue
        d_arr, F_arr = rep_curves[grp]
        lw = 2.4 if grp in ('Ax_45', 'Ax_60') else 1.6
        ax.plot(d_arr, F_arr, color=GPAL[grp], linewidth=lw,
                label=GLABELS[grp],
                zorder=6 if grp == 'Ax_45' else 3)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlim(left=-0.02); ax.set_ylim(bottom=-30)
    polish(ax, xlabel='Displacement (mm)', ylabel='Force (N)')
    leg = ax.legend(title='Group', title_fontsize=9, loc='upper left',
                    frameon=True, edgecolor='black', fancybox=False,
                    ncol=2, columnspacing=0.8, fontsize=8.5)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 1 -- Representative F-D curves (illustrative, extra)")

    # ── Fig 2: Peak Force by Group (bar + strip + CLD) ────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng2  = np.random.RandomState(42)
    pos   = np.arange(len(GROUP_ORDER))
    y_max = []
    for i, grp in enumerate(GROUP_ORDER):
        sub = df[df['group']==grp]['max_force_N'].values
        mn, sd = sub.mean(), sub.std(ddof=1)
        ax.bar(i, mn, width=0.55, color=GPAL[grp], alpha=0.82,
               edgecolor='black', linewidth=0.8, zorder=3)
        ax.errorbar(i, mn, yerr=sd, fmt='none', color='black',
                    linewidth=1.2, capsize=4, capthick=1.2, zorder=5)
        jit = rng2.uniform(-0.15, 0.15, len(sub))
        ax.scatter(i+jit, sub, color=GPAL[grp], edgecolors='black',
                   linewidths=0.6, s=22, zorder=6, alpha=0.9)
        y_max.append(sub.max())
    for i, grp in enumerate(GROUP_ORDER):
        sub = df[df['group']==grp]['max_force_N'].values
        mn, sd = sub.mean(), sub.std(ddof=1)
        ax.text(i, mn+sd+55, cld_force[grp], ha='center', va='bottom',
                fontsize=11, color='black', fontfamily=TNR)
    base_mean = df[df['group']=='Base']['max_force_N'].mean()
    ax.axhline(base_mean, color='#555555', linewidth=1.0, linestyle='--',
               alpha=0.7, zorder=2, label='Base mean')
    ax.set_xticks(pos)
    ax.set_xticklabels([GLABELS[g] for g in GROUP_ORDER], fontsize=10)
    ax.set_ylim([max(0, min(y_max)*0.5), max(y_max)+220])
    polish(ax, ylabel='Peak Lap-Shear Force (N)')
    ax.legend(frameon=True, edgecolor='black', fancybox=False,
              loc='upper right', fontsize=8.5).get_frame().set_linewidth(0.8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 2 -- Peak Force by Group")

    # ── Fig 3: Fracture Energy by Group ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng3  = np.random.RandomState(43)
    y_max_e = []
    for i, grp in enumerate(GROUP_ORDER):
        sub = df[df['group']==grp]['energy_Nmm'].values
        mn, sd = sub.mean(), sub.std(ddof=1)
        ax.bar(i, mn, width=0.55, color=GPAL[grp], alpha=0.82,
               edgecolor='black', linewidth=0.8, zorder=3)
        ax.errorbar(i, mn, yerr=sd, fmt='none', color='black',
                    linewidth=1.2, capsize=4, capthick=1.2, zorder=5)
        jit = rng3.uniform(-0.15, 0.15, len(sub))
        ax.scatter(i+jit, sub, color=GPAL[grp], edgecolors='black',
                   linewidths=0.6, s=22, zorder=6, alpha=0.9)
        y_max_e.append(sub.max())
    for i, grp in enumerate(GROUP_ORDER):
        sub = df[df['group']==grp]['energy_Nmm'].values
        mn, sd = sub.mean(), sub.std(ddof=1)
        ax.text(i, mn+sd+40, cld_energy[grp], ha='center', va='bottom',
                fontsize=11, color='black', fontfamily=TNR)
    base_E = df[df['group']=='Base']['energy_Nmm'].mean()
    ax.axhline(base_E, color='#555555', linewidth=1.0, linestyle='--',
               alpha=0.7, zorder=2, label='Base mean')
    ax.set_xticks(pos)
    ax.set_xticklabels([GLABELS[g] for g in GROUP_ORDER], fontsize=10)
    ax.set_ylim([max(0, min(y_max_e)*0.5), max(y_max_e)+200])
    polish(ax, ylabel='Fracture Energy (N·mm)')
    ax.legend(frameon=True, edgecolor='black', fancybox=False,
              loc='upper right', fontsize=8.5).get_frame().set_linewidth(0.8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 3 -- Fracture Energy by Group")

    # ── Fig 4: Parametric Effects 3-panel (merged old Figs 4+5) ───
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    il_vals  = [0.6, 0.8, 1.0]
    LS       = ['-', '--', ':']
    MARKERS  = ['o', 's', '^']
    ref_styles = [('FGax_T1','FGax-T1','--'),
                  ('FGax_T2','FGax-T2','-.'),
                  ('Base',   'Base',   ':'),
                  ('Flat',   'Flat',   (0,(5,2,1,2)))]

    for axi, (col, ytitle) in enumerate([('max_force_N','Peak Force (N)'),
                                          ('energy_Nmm', 'Fracture Energy (N·mm)')]):
        ax = axes[axi]
        for rg, rl, rst in ref_styles:
            rm = df[df['group']==rg][col].mean()
            ax.axhline(rm, color=GPAL[rg], linewidth=0.9, linestyle=rst,
                       alpha=0.65, label=rl, zorder=2)
        for grp, ls, mk in zip(['Ax_45','Ax_60','Ax_75'], LS, MARKERS):
            means, sds = [], []
            for il in il_vals:
                sub = df[(df['group']==grp)&(abs(df['interlayer_mm']-il)<0.01)][col]
                means.append(sub.mean()); sds.append(sub.std(ddof=1))
            means, sds = np.array(means), np.array(sds)
            ax.plot(il_vals, means, color=GPAL[grp], linewidth=1.8,
                    linestyle=ls, marker=mk, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.7,
                    label=GLABELS[grp], zorder=5)
            ax.fill_between(il_vals, means-sds, means+sds,
                            color=GPAL[grp], alpha=0.10)
        ax.set_xticks(il_vals)
        polish(ax, xlabel='Interlayer Thickness (mm)', ylabel=ytitle)

    ax = axes[2]
    st_vals = [0.6, 0.8, 1.0]
    IL_C  = {1.0: IL_COLORS_DS[1.0], 0.8: IL_COLORS_DS[0.8], 0.6: IL_COLORS_DS[0.6]}
    IL_LS = {1.0: '-', 0.8: '--', 0.6: ':'}
    IL_MK = {1.0: 'o', 0.8: 's', 0.6: '^'}
    for il in [1.0, 0.8, 0.6]:
        means = []
        for st in st_vals:
            sub = df[(df['interlayer_mm'].round(1)==il)&
                     (df['strut_mm'].round(1)==st)&
                     (df['group'].str.startswith('Ax'))]['max_force_N']
            means.append(sub.mean() if len(sub) else np.nan)
        ax.plot(st_vals, means, color=IL_C[il], linewidth=1.8,
                linestyle=IL_LS[il], marker=IL_MK[il], markersize=7,
                markeredgecolor='black', markeredgewidth=0.7,
                label=f'IL = {il} mm', zorder=5)
    ax.set_xticks(st_vals)
    polish(ax, xlabel='Strut Length / Height (mm)', ylabel='Mean Peak Force (N)')

    handles_AB, labels_AB = axes[0].get_legend_handles_labels()
    handles_C,  labels_C  = axes[2].get_legend_handles_labels()
    seen = set(); handles_all = []; labels_all = []
    for h, l in zip(handles_AB + handles_C, labels_AB + labels_C):
        if l not in seen:
            seen.add(l); handles_all.append(h); labels_all.append(l)
    fig.legend(handles_all, labels_all, loc='lower center', ncol=6,
               frameon=True, edgecolor='black', fancybox=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 4 -- Parametric effects 3-panel (merged IL + strut)")

    # ── Fig 5-code: Cohen's d Forest Plot ──────────────────────────
    # (Numbered "Fig 5" in the v9.0 script; the paper's own Fig. 5 is the
    # AI-threshold scatter, produced separately below in Part B2/THRESHOLD_PDF.)
    fig, axes5 = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for axi, (col, label) in enumerate([('max_force_N', 'Peak Force (N)'),
                                         ('energy_Nmm',  'Fracture Energy (N·mm)')]):
        ax = axes5[axi]
        sub = df_cohen[df_cohen['col']==col].sort_values('d')
        y   = np.arange(len(sub))
        colors_d = [GPAL.get(g, '#888') for g in sub['group']]
        ax.barh(y, sub['d'], height=0.55, color=colors_d, alpha=0.85,
                edgecolor='black', linewidth=0.8, zorder=3)
        ax.errorbar(sub['d'], y,
                    xerr=[sub['d']-sub['CI_lo'], sub['CI_hi']-sub['d']],
                    fmt='none', color='black', linewidth=1.1,
                    capsize=3.5, capthick=1.1, zorder=5)
        ax.axvline(0, color='black', linewidth=0.9, zorder=4)
        for xv, ls in [(-0.8,'--'),(-0.5,':'),(0.5,':'),(0.8,'--')]:
            ax.axvline(xv, color='#999', linewidth=0.7, linestyle=ls,
                       alpha=0.6, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([GLABELS[g] for g in sub['group']], fontsize=9.5)
        polish(ax, xlabel="Cohen's d  (vs. Base)")
        ax.set_title(label, fontsize=10, pad=5, fontfamily=TNR)
        for yi, (_, row) in enumerate(sub.iterrows()):
            short = {'very large':'VL','large':'Lg','medium':'Med',
                     'small':'Sm','negligible':'ns'}.get(row['mag'], '?')
            xpos = row['CI_lo'] - 0.10
            ax.text(xpos, yi, short, va='center', ha='right', fontsize=7.5,
                    fontfamily=TNR, color='#333333',
                    bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.5))
    thr_lines = [Line2D([0],[0], color='#999', lw=0.7, ls='--',
                         label='|d| = 0.8 (large)'),
                 Line2D([0],[0], color='#999', lw=0.7, ls=':',
                         label='|d| = 0.5 (medium)')]
    axes5[1].legend(handles=thr_lines, loc='lower right', frameon=True,
                    edgecolor='black', fancybox=False,
                    fontsize=8).get_frame().set_linewidth(0.8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 5-code -- Cohen's d forest plot")

    # ── Fig 6: Failure Mode Stacked Bar ───────────────────────────
    MODE_COLORS = {'adhesive':'#D55E00','adhesive_mixed':'#E69F00',
                   'cohesive':'#009E73','mixed':'#0072B2',
                   'cohesive_mixed':'#66C2A5','cohesive_interlayer':'#1D3557',
                   'mixed_buckling':'#4682B4'}
    fig, ax = plt.subplots(figsize=(9, 5))
    ct_pct = ct.div(ct.sum(axis=1), axis=0)*100
    ct_pct = ct_pct.reindex(GROUP_ORDER).fillna(0)
    modes_present = list(ct_pct.columns)
    bottoms = np.zeros(len(GROUP_ORDER))
    for mode in modes_present:
        vals = ct_pct[mode].reindex(GROUP_ORDER).fillna(0).values
        mkey = str(mode).lower().replace(' ', '_').replace('-', '_')
        ax.bar(range(len(GROUP_ORDER)), vals, bottom=bottoms,
               color=MODE_COLORS.get(mkey, '#AAAAAA'), alpha=0.88, width=0.6,
               edgecolor='black', linewidth=0.8,
               label=str(mode).replace('_',' ').title(), zorder=3)
        for bi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 18:
                ax.text(bi, b+v/2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=9.5, color='white')
        bottoms += vals
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GLABELS[g] for g in GROUP_ORDER], fontsize=10)
    ax.set_ylim([0, 112]); ax.set_yticks([0, 25, 50, 75, 100])
    polish(ax, ylabel='Proportion (%)')
    ax.text(0.01, 0.99,
            f"chi2={chi2_s:.1f}, df={dof_chi}, p<0.001, Cramer's V={cramv:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            fontfamily=TNR,
            bbox=dict(fc='white', ec='black', lw=0.7, boxstyle='round,pad=0.25'))
    handles6, labels6 = ax.get_legend_handles_labels()
    fig.legend(handles6, labels6, loc='lower center', ncol=4,
               frameon=True, edgecolor='black', fancybox=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 6 -- Failure mode distribution")

    # ── Fig 7: Feature-Target Correlation Heatmap ─────────────────
    feat_cols_c = ['auxetic_index','interlayer_mm','strut_mm','porosity_pct',
                   'grading_ratio','cell_aspect_ratio','epoxy_thickness_mm']
    tgt_cols_c  = ['max_force_N','energy_Nmm','SEA_Nmm_mm3',
                   'stiffness_N_mm','displacement_mm']
    tgt_cols_c  = [c for c in tgt_cols_c if c in df.columns]
    feat_labels_c = ['Auxetic Index','Interlayer (mm)','Strut Length (mm)',
                     'Porosity (%)','Grading Ratio','Cell Aspect Ratio',
                     'Epoxy Thickness (mm)']
    tgt_labels_full = {'max_force_N':'Peak\nForce', 'energy_Nmm':'Fracture\nEnergy',
                        'SEA_Nmm_mm3':'SEA', 'stiffness_N_mm':'Stiffness',
                        'displacement_mm':'Displacement'}
    tgt_labels_c  = [tgt_labels_full[c] for c in tgt_cols_c]
    corr_mat = np.zeros((len(feat_cols_c), len(tgt_cols_c)))
    p_mat    = np.zeros_like(corr_mat)
    for fi, fc in enumerate(feat_cols_c):
        for ti, tc in enumerate(tgt_cols_c):
            if fc in df.columns and tc in df.columns:
                r, p = stats.pearsonr(df[fc], df[tc])
                corr_mat[fi, ti] = round(r, 3); p_mat[fi, ti] = p
    fig, ax = plt.subplots(figsize=(8, 6))
    div_cmap = LinearSegmentedColormap.from_list(
        'divBlRd', [(0,'#D55E00'),(0.5,'white'),(1,'#0072B2')])
    im2 = ax.imshow(corr_mat, cmap=div_cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tgt_cols_c)))
    ax.set_xticklabels(tgt_labels_c, fontsize=9.5, rotation=0, ha='center')
    ax.set_yticks(range(len(feat_cols_c)))
    ax.set_yticklabels(feat_labels_c, fontsize=9.5)
    for fi in range(len(feat_cols_c)):
        for ti in range(len(tgt_cols_c)):
            r = corr_mat[fi, ti]; p = p_mat[fi, ti]
            star = ('***' if p<0.001 else '**' if p<0.01
                    else '*' if p<0.05 else '')
            fc_txt = 'white' if abs(r) > 0.5 else 'black'
            ax.text(ti, fi, f'{r:.2f}{star}', ha='center', va='center',
                    fontsize=9, color=fc_txt)
    for sp in ax.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(0.9)
    ax.tick_params(axis='both', colors='black', width=0.9, length=4)
    cb7 = plt.colorbar(im2, ax=ax, fraction=0.035, pad=0.04)
    cb7.set_label('Pearson r', fontsize=10, fontfamily=TNR)
    cb7.outline.set_edgecolor('black'); cb7.outline.set_linewidth(0.9)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 7 -- Feature-target correlation heatmap")

print(f"  Analysis figures -> {ANALYSIS_PDF}")

# ── Supplementary PDF: Tukey HSD significance matrix ─────────────────────────
print("\nGenerating supplementary PDF...")
with PdfPages(SUPP_PDF) as pdf_s:
    fig_s, ax_s = plt.subplots(figsize=(8, 6.5))
    n = len(GROUP_ORDER)
    mat = np.ones((n, n))
    tdf_F = tukey_all['max_force_N']
    for _, r in tdf_F.iterrows():
        if r.A in GROUP_ORDER and r.B in GROUP_ORDER:
            i, j = GROUP_ORDER.index(r.A), GROUP_ORDER.index(r.B)
            mat[i, j] = r.p; mat[j, i] = r.p
    cmap_s = LinearSegmentedColormap.from_list(
        'pval', [(0,'#0072B2'),(0.05,'#A8D8F0'),(0.051,'#F0F0F0'),(1,'#F0F0F0')])
    im_s = ax_s.imshow(mat, cmap=cmap_s, vmin=0, vmax=0.12, aspect='equal')
    labels_xy = [GLABELS[g] for g in GROUP_ORDER]
    ax_s.set_xticks(range(n))
    ax_s.set_xticklabels(labels_xy, fontsize=9.5, rotation=45, ha='right')
    ax_s.set_yticks(range(n)); ax_s.set_yticklabels(labels_xy, fontsize=9.5)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax_s.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                               fill=True, facecolor='#D0D0D0',
                               edgecolor='black', lw=0.5, zorder=3))
                continue
            p = mat[i, j]
            sig_str = ('***' if p<0.001 else '**' if p<0.01
                       else '*' if p<0.05 else 'ns')
            fc_t = 'white' if p < 0.05 else '#555'
            ax_s.text(j, i, sig_str, ha='center', va='center',
                      fontsize=8.5, color=fc_t)
    cb_s = plt.colorbar(im_s, ax=ax_s, fraction=0.035, pad=0.04)
    cb_s.set_label('p-value', fontsize=10, fontfamily=TNR)
    cb_s.outline.set_edgecolor('black'); cb_s.outline.set_linewidth(0.9)
    for sp in ax_s.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(0.9)
    ax_s.set_title('Tukey HSD Pairwise Significance Matrix (Peak Force)\n'
                   '(blue = p<0.05; *** p<0.001; ** p<0.01; * p<0.05; ns = not significant)',
                   fontsize=9.5, pad=8, fontfamily=TNR)
    fig_s.tight_layout()
    pdf_s.savefig(fig_s, bbox_inches='tight'); plt.close(fig_s)
print(f"  Supplementary -> {SUPP_PDF}")


# ── PART B2: AI ACTIVATION-THRESHOLD BOOTSTRAP (Sec. 2.6, 3.1, Fig. 5) ───────
# Story: Section 3.1 makes a sharp empirical claim -- "a measurable Fmax gain
# over Base requires AI >= 0.40" -- and backs it with a 95% bootstrap CI
# [0.225, 0.414] on a piecewise-linear breakpoint. None of that appeared in
# v9.0. This block reproduces it: on each of 10,000 bootstrap resamples of
# the 160 training specimens, fit a two-segment piecewise-linear Fmax(AI)
# model (flat-then-rising, breakpoint located by grid search over
# AI in [0, 0.50]) and record where the breakpoint falls.
print("\nBootstrapping the AI activation threshold (Sec. 2.6 / 3.1)...")

def _piecewise_sse(ai, y, bp):
    """SSE of the best piecewise-linear (bent-line) fit with a breakpoint at
    bp: two independent linear segments joined continuously at bp."""
    left  = ai <= bp
    right = ~left
    if left.sum() < 2 or right.sum() < 2:
        return np.inf
    # Fit two OLS lines constrained to meet at (bp, level) via a basis of
    # [1, ai, max(0, ai-bp)] -- equivalent to a continuous piecewise-linear
    # (hinge) regression, which is the standard "single breakpoint" model.
    hinge = np.clip(ai - bp, 0, None)
    Xd = np.column_stack([np.ones_like(ai), ai, hinge])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return float(np.sum(resid**2))

def fit_breakpoint(ai, y, grid):
    sses = [_piecewise_sse(ai, y, bp) for bp in grid]
    return grid[int(np.argmin(sses))]

_bp_grid = np.linspace(0.00, 0.50, 101)
ai_arr = df['auxetic_index'].values.astype(float)
F_arr  = df['max_force_N'].values.astype(float)
best_bp_full = fit_breakpoint(ai_arr, F_arr, _bp_grid)

rng_bs = np.random.RandomState(42)
n_spec = len(df)
boot_bps = np.empty(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    idx = rng_bs.randint(0, n_spec, n_spec)
    boot_bps[b] = fit_breakpoint(ai_arr[idx], F_arr[idx], _bp_grid)
ci_lo, ci_hi = np.percentile(boot_bps, [2.5, 97.5])
print(f"  Breakpoint (full sample): AI* = {best_bp_full:.3f}")
print(f"  95% bootstrap CI ({N_BOOTSTRAP} resamples): [{ci_lo:.3f}, {ci_hi:.3f}]")

df_bootstrap = pd.DataFrame({
    'n_resamples': [N_BOOTSTRAP], 'breakpoint_full_sample': [round(best_bp_full,4)],
    'CI_2.5%': [round(float(ci_lo),4)], 'CI_97.5%': [round(float(ci_hi),4)],
})

# ── Fig 5 (paper numbering): AI-threshold scatter with bootstrap band ────────
print("Generating AI-threshold figure...")
with PdfPages(THRESHOLD_PDF) as pdf_t:
    fig_t, ax_t = plt.subplots(figsize=(9, 5.5))
    for grp in GROUP_ORDER:
        sub = df[df['group']==grp]
        if not len(sub): continue
        ax_t.scatter(sub['auxetic_index'], sub['max_force_N'], color=GPAL[grp],
                     s=32, alpha=0.85, edgecolors='black', linewidths=0.4,
                     label=GLABELS[grp], zorder=5)
    ax_t.axvspan(ci_lo, ci_hi, color='#D55E00', alpha=0.12, zorder=1,
                 label=f'95% bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax_t.axvline(best_bp_full, color='#D55E00', linewidth=1.4, linestyle=':',
                 zorder=4, label=f'Threshold AI* = {best_bp_full:.2f}')
    base_mean = df[df['group']=='Base']['max_force_N'].mean()
    ax_t.axhline(base_mean, color='#555', linewidth=1.0, linestyle='--',
                 alpha=0.7, zorder=2, label='Base mean')
    polish(ax_t, xlabel='Auxeticity Index (AI)', ylabel='Peak Force, Fmax (N)')
    ax_t.legend(loc='lower right', ncol=2, frameon=True, edgecolor='black',
                fancybox=False, fontsize=8).get_frame().set_linewidth(0.8)
    fig_t.tight_layout()
    pdf_t.savefig(fig_t, bbox_inches='tight'); plt.close(fig_t)
print(f"  AI-threshold figure -> {THRESHOLD_PDF}")


# ── PART D: ML PIPELINE ───────────────────────────────────────────────────────
# Story: can a handful of geometric parameters predict joint strength well
# enough to trust for inverse design? Sec. 4.1 trains five surrogate model
# classes on Fmax under a *configuration-grouped* 5-fold scheme (so all four
# replicates of one geometry stay together, never split across train/test);
# this script's ML section additionally carries three extra targets
# (energy_Nmm, SEA_Nmm_mm3, stiffness_N_mm) through the same machinery as a
# bonus beyond what the paper reports.
print("\nRunning ML pipeline...")

from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GroupKFold, StratifiedKFold, cross_val_score,
                                      cross_val_predict, LeaveOneGroupOut,
                                      RandomizedSearchCV)
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance
from scipy.stats import norm as spn

try:
    import xgboost as xgb; XGB = True
except ImportError:
    XGB = False
    print("  xgboost not found -- GradientBoosting fallback")
try:
    import shap; SHAP = True
except ImportError:
    SHAP = False
    print("  shap not found -- permutation importance fallback")

FEATURES = ['auxetic_index','interlayer_mm','strut_mm','porosity_pct',
            'grading_ratio','cell_aspect_ratio','epoxy_thickness_mm',
            'l_abs_mm','cell_pitch_x_mm']
# Sec. 4.1: "For reference configurations lacking explicit cellular geometries
# (Base, Flat, Spacer), the null geometric inputs (AI, strut thickness,
# porosity) were zero-imputed." Applied broadly across all FEATURES here as a
# defensive measure in case the source spreadsheet leaves any of them blank
# for those three families, in addition to the original l_abs/cell_pitch fill.
df[FEATURES] = df[FEATURES].fillna(0.0)
FEAT_LABELS = {
    'auxetic_index':       'Auxetic Index (AI)',
    'interlayer_mm':       'Interlayer (mm)',
    'strut_mm':            'Strut Thickness (mm)',
    'porosity_pct':        'Porosity (%)',
    'grading_ratio':       'Grading Ratio',
    'cell_aspect_ratio':   'Cell Aspect Ratio',
    'epoxy_thickness_mm':  'Epoxy Thickness (mm)',
    'l_abs_mm':            'Strut Length ℓ (mm, back-calc)',
    'cell_pitch_x_mm':     'Cell Pitch, load-dir (mm)',
}
TARGETS = {
    'max_force_N':     'Peak Force (N)',
    'energy_Nmm':      'Fracture Energy (N·mm)',
    'SEA_Nmm_mm3':     'SEA (N·mm/mm³)',
    'stiffness_N_mm':  'Stiffness (N/mm)',
}
TARGETS = {k: v for k, v in TARGETS.items() if k in df.columns}
X       = df[FEATURES].values
y_dict  = {c: df[c].values for c in TARGETS}
tgt_list = list(TARGETS.keys())
PRIMARY_TARGET = 'max_force_N'   # the paper's single ML target (Sec. 4.1)
groups_logo = df['configuration'].values

_raw_classes = sorted(df['failure_mode'].unique())
_cls_map     = {c: i for i, c in enumerate(_raw_classes)}
y_cls        = np.array([_cls_map[v] for v in df['failure_mode']], dtype=int)
cls_names    = [str(c) for c in _raw_classes]
le           = LabelEncoder()
le.classes_  = np.array(_raw_classes)
print(f"  y_cls unique values: {np.unique(y_cls).tolist()}  ->  classes: {cls_names}")

RAND = 42
# FIXED (Sec. 4.1): "configuration-grouped 5-fold scheme... GroupKFold...
# All four physical replicates of any given configuration were assigned
# exclusively to the same fold." v9.0 used a plain shuffled KFold here, which
# does not respect that grouping and can leak replicates across train/test.
gkf5 = GroupKFold(n_splits=5)
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND)


# ── Hyperparameter search spaces (Sec. 4.1: "3-fold randomized search... the
# final architecture... reflects the across-fold mode of this search") ──────
# NOTE: the exact ranges live in the paper's Supplementary Table S1, which
# was not available when this script was prepared. The ranges below are
# reasonable, literature-typical spans for each model class -- swap them for
# the real Table S1 values if/when you have that file; the tuning *procedure*
# (nested RandomizedSearchCV per outer fold, final model = across-fold mode
# of the winning hyperparameters) already matches Sec. 4.1 either way.
PARAM_DISTRIBUTIONS = {
    'XGBoost': dict(
        n_estimators=[200, 300, 400, 600], max_depth=[3, 4, 5, 6],
        learning_rate=[0.02, 0.04, 0.06, 0.08, 0.10], subsample=[0.6, 0.8, 1.0],
        colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 3, 5],
        reg_alpha=[0.0, 0.1, 0.5], reg_lambda=[0.5, 1.0, 2.0]),
    'RandomForest': dict(
        n_estimators=[200, 300, 400, 600], max_depth=[4, 6, 8, 10, None],
        min_samples_leaf=[1, 2, 3, 5], max_features=['sqrt', 0.5, 1.0]),
    'GPR': dict(alpha=[1e-4, 1e-3, 1e-2], n_restarts_optimizer=[3, 5, 8]),
    'SVR': dict(C=[10, 50, 100, 200, 500], gamma=['scale', 'auto', 0.01, 0.1],
                epsilon=[0.01, 0.03, 0.05, 0.1]),
    'Ridge': dict(alpha=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
}

def make_xgb_reg(**kw):
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                  gamma=0.1, reg_alpha=0.1, reg_lambda=1.0)
    params.update(kw)
    if XGB:
        return xgb.XGBRegressor(random_state=RAND, verbosity=0, **params)
    params.pop('colsample_bytree', None); params.pop('min_child_weight', None)
    params.pop('gamma', None); params.pop('reg_alpha', None); params.pop('reg_lambda', None)
    return GradientBoostingRegressor(random_state=RAND, min_samples_leaf=3, **params)

def make_rf_reg(**kw):
    params = dict(n_estimators=400, max_depth=8, min_samples_leaf=2)
    params.update(kw)
    return RandomForestRegressor(random_state=RAND, **params)

def make_gpr_reg(**kw):
    alpha = kw.pop('alpha', 1e-3)
    n_restarts = kw.pop('n_restarts_optimizer', 5)
    kernel = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5)+WhiteKernel(alpha)
    return Pipeline([('sc', StandardScaler()),
                      ('gpr', GaussianProcessRegressor(
                          kernel=kernel, n_restarts_optimizer=n_restarts,
                          normalize_y=True, random_state=RAND))])

def make_svr_reg(**kw):
    params = dict(kernel='rbf', C=200, gamma='scale', epsilon=0.03)
    params.update(kw)
    return Pipeline([('sc', StandardScaler()), ('svr', SVR(**params))])

def make_ridge_reg(**kw):
    params = dict(alpha=1.0); params.update(kw)
    return Pipeline([('sc', StandardScaler()), ('ridge', Ridge(**params))])

MODEL_FACTORY = {'XGBoost': make_xgb_reg, 'RandomForest': make_rf_reg,
                  'GPR': make_gpr_reg, 'SVR': make_svr_reg, 'Ridge': make_ridge_reg}

def tune_across_folds(model_name, X, y, groups, n_splits=5, n_iter=N_ITER_SEARCH,
                       inner_cv=INNER_CV_FOLDS, random_state=RAND):
    """Sec. 4.1 tuning protocol: for each of the 5 outer GroupKFold training
    folds, run an `inner_cv`-fold RandomizedSearchCV over PARAM_DISTRIBUTIONS
    and record the winning hyperparameters; the final architecture is the
    across-fold *mode* (most frequent winning value) of each hyperparameter.
    Returns (best_params_dict, per_fold_params_list)."""
    factory = MODEL_FACTORY[model_name]
    dist = PARAM_DISTRIBUTIONS[model_name]
    gkf = GroupKFold(n_splits=n_splits)
    fold_params = []
    for tr_idx, _ in gkf.split(X, y, groups):
        Xtr, ytr, gtr = X[tr_idx], y[tr_idx], groups[tr_idx]
        n_inner = min(inner_cv, len(np.unique(gtr)))
        if n_inner < 2:
            continue
        base = factory()
        # RandomizedSearchCV needs a plain estimator name -> param prefix when
        # the model is wrapped in a Pipeline (GPR/SVR/Ridge); handle both.
        if isinstance(base, Pipeline):
            step_name = base.steps[-1][0]
            pdist = {f'{step_name}__{k}': v for k, v in dist.items()}
        else:
            pdist = dist
        search = RandomizedSearchCV(base, pdist, n_iter=min(n_iter, 20),
                                     cv=GroupKFold(n_splits=n_inner),
                                     scoring='r2', random_state=random_state,
                                     n_jobs=1)
        try:
            search.fit(Xtr, ytr, groups=gtr)
            fold_params.append(search.best_params_)
        except Exception as e:
            print(f"    [tune:{model_name}] fold skipped ({e})")
    if not fold_params:
        return {}, []
    # across-fold mode, per hyperparameter
    keys = fold_params[0].keys()
    mode_params = {}
    for k in keys:
        vals = [str(fp.get(k)) for fp in fold_params]
        winner = Counter(vals).most_common(1)[0][0]
        # recover the original (typed) value from whichever fold produced it
        mode_params[k] = next(fp[k] for fp in fold_params if str(fp[k]) == winner)
    return mode_params, fold_params

print("  Tuning surrogate hyperparameters (across-fold mode, Sec. 4.1)...")
print(f"    [{N_ITER_SEARCH} draws/fold x {INNER_CV_FOLDS}-fold inner CV -- "
      f"this is the slowest step in the script; lower N_ITER_SEARCH for a quick run]")
TUNED_PARAMS = {}
for mn in MODEL_FACTORY:
    best, _ = tune_across_folds(mn, X, y_dict[PRIMARY_TARGET], groups_logo)
    TUNED_PARAMS[mn] = best
    print(f"    {mn}: {best}")

def build_tuned(model_name):
    factory = MODEL_FACTORY[model_name]
    params = TUNED_PARAMS.get(model_name, {})
    if not params:
        return factory()
    # unwrap 'stepname__param' keys back to plain kwargs for the factory
    step_prefix = None
    probe = factory()
    if isinstance(probe, Pipeline):
        step_prefix = probe.steps[-1][0] + '__'
    clean = {(k[len(step_prefix):] if step_prefix and k.startswith(step_prefix) else k): v
             for k, v in params.items()}
    return factory(**clean)

REG_MODELS = {mn: build_tuned(mn) for mn in MODEL_FACTORY}

CLS_MODELS = {
    'XGBoost':      GradientBoostingClassifier(   # drop-in, no label restriction
                        n_estimators=300, max_depth=4,
                        learning_rate=0.05, subsample=0.8,
                        random_state=RAND),
    'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=6,
                        min_samples_leaf=2, random_state=RAND),
}


# ── Configuration-grouped 5-fold CV (the paper's primary metric, Fig. 8/Table 4)
cv_rows = []
print("  5-fold GroupKFold CV regression...")
for mn, model in REG_MODELS.items():
    for col, label in TARGETS.items():
        y = y_dict[col]
        try:
            r2s = cross_val_score(model, X, y, cv=gkf5, groups=groups_logo, scoring='r2')
            mas = -cross_val_score(model, X, y, cv=gkf5, groups=groups_logo,
                                    scoring='neg_mean_absolute_error')
            rms = np.sqrt(-cross_val_score(model, X, y, cv=gkf5, groups=groups_logo,
                                            scoring='neg_mean_squared_error'))
            cv_rows.append(dict(model=mn, target=label, col=col, cv='5-fold-GroupKFold',
                                R2_mean=round(r2s.mean(),4),
                                R2_std=round(r2s.std(),4),
                                MAE_mean=round(mas.mean(),3),
                                MAE_std=round(mas.std(),3),
                                RMSE_mean=round(rms.mean(),3)))
        except Exception as e:
            print(f"    {mn}/{col}: {e}")

# ── LOGO-CV (kept as extra beyond the paper's own "5-fold" GroupKFold metric —
# a stricter, "generalize to a totally unseen configuration" stress test) ────
print("  Leave-One-Group-Out CV (extra, stricter than the paper's 5-fold)...")
logo = LeaveOneGroupOut()
for mn in ['XGBoost', 'RandomForest']:
    model = REG_MODELS[mn]
    for col, label in TARGETS.items():
        y  = y_dict[col]
        yp = cross_val_predict(model, X, y, cv=logo, groups=groups_logo)
        r2v = r2_score(y, yp); mav = mean_absolute_error(y, yp)
        rmsv = np.sqrt(mean_squared_error(y, yp))
        cv_rows.append(dict(model=mn, target=label, col=col, cv='LOGO',
                            R2_mean=round(r2v,4), R2_std=np.nan,
                            MAE_mean=round(mav,3), MAE_std=np.nan,
                            RMSE_mean=round(rmsv,3)))
df_cv = pd.DataFrame(cv_rows)

# ── Failure-mode classification (extra -- not discussed in the paper, which
# models Fmax only; kept because it's a reasonable bonus use of the same
# feature set, just clearly labeled as outside the paper's own scope) ───────
print("  Classification (extra, not in paper)...")
cls_rows = []
for mn, model in CLS_MODELS.items():
    acc  = cross_val_score(model, X, y_cls, cv=skf5, scoring='accuracy')
    f1w  = cross_val_score(model, X, y_cls, cv=skf5, scoring='f1_weighted')
    yp_cls = cross_val_predict(model, X, y_cls, cv=skf5)
    f1_per = f1_score(y_cls, yp_cls, average=None)
    r = {'model': mn, 'Acc_mean': round(acc.mean(),4), 'Acc_std': round(acc.std(),4),
         'F1w_mean': round(f1w.mean(),4), 'F1w_std': round(f1w.std(),4)}
    for ci, cn in enumerate(cls_names):
        r[f'F1_{cn}'] = round(float(f1_per[ci]), 4)
    cls_rows.append(r)
df_cls = pd.DataFrame(cls_rows)

# ── Fit final models (on all 160 training specimens) + overfitting-gap check ─
print("  Fitting final models + train-vs-CV R^2 overfitting check (Sec. 4.3)...")
final = {}
overfit_rows = []
for col in TARGETS:
    y = y_dict[col]
    xm = build_tuned('XGBoost'); xm.fit(X, y)
    rm = build_tuned('RandomForest'); rm.fit(X, y)
    gm = build_tuned('GPR'); gm.fit(X, y)   # Pipeline: scales internally
    final[col] = {'XGBoost': xm, 'RF': rm, 'GPR': gm}
    train_r2_xgb = r2_score(y, xm.predict(X))
    train_r2_gpr = r2_score(y, gm.predict(X))
    cv_r2_xgb = df_cv[(df_cv['model']=='XGBoost') & (df_cv['col']==col) &
                       (df_cv['cv']=='5-fold-GroupKFold')]['R2_mean']
    cv_r2_gpr = df_cv[(df_cv['model']=='GPR') & (df_cv['col']==col) &
                       (df_cv['cv']=='5-fold-GroupKFold')]['R2_mean']
    overfit_rows.append(dict(target=TARGETS[col], col=col,
                              XGBoost_train_R2=round(train_r2_xgb,4),
                              XGBoost_CV_R2=float(cv_r2_xgb.iloc[0]) if len(cv_r2_xgb) else np.nan,
                              GPR_train_R2=round(train_r2_gpr,4),
                              GPR_CV_R2=float(cv_r2_gpr.iloc[0]) if len(cv_r2_gpr) else np.nan))
df_overfit = pd.DataFrame(overfit_rows)
if PRIMARY_TARGET in TARGETS:
    _r = df_overfit[df_overfit['col']==PRIMARY_TARGET].iloc[0]
    print(f"    Fmax: XGBoost train R2={_r['XGBoost_train_R2']:.3f} vs CV R2="
          f"{_r['XGBoost_CV_R2']:.3f}  |  GPR train R2={_r['GPR_train_R2']:.3f} "
          f"vs CV R2={_r['GPR_CV_R2']:.3f}")

# ── Ensemble weights (Sec. 4.1): "weighted inversely proportional to their
# 5-fold cross-validated RMSE values" — computed from df_cv rather than
# hardcoded, so it re-derives itself if you re-run with real data. ──────────
def ensemble_weights(col, models=('XGBoost', 'GPR')):
    rmses = {}
    for mn in models:
        row = df_cv[(df_cv['model']==mn) & (df_cv['col']==col) &
                    (df_cv['cv']=='5-fold-GroupKFold')]
        rmses[mn] = float(row['RMSE_mean'].iloc[0]) if len(row) else np.nan
    inv = {mn: 1.0/r for mn, r in rmses.items() if r and not np.isnan(r) and r > 0}
    s = sum(inv.values())
    return {mn: inv.get(mn, 0.0)/s for mn in models} if s > 0 else \
           {mn: 1.0/len(models) for mn in models}

ENSEMBLE_W = ensemble_weights(PRIMARY_TARGET)
print(f"  Ensemble weights for {PRIMARY_TARGET} (Sec. 4.1): {ENSEMBLE_W}")


# ── Feature importance + SHAP (XGBoost AND RandomForest, Sec. 4.3) ──────────
# ADDED: v9.0 only ran TreeExplainer on XGBoost. Sec. 4.3 explicitly reports a
# second SHAP pass on Random Forest "as a robustness check" and states the
# top-3 feature ranking (AI > porosity > IL) is identical between the two --
# that comparison can't be made without computing the RF SHAP values too.
print("  Feature importance / SHAP (XGBoost + RandomForest)...")
fi_rows = []; shap_rows = []; shap_vals_all = {}; shap_vals_rf_all = {}
feat_names = FEATURES
for col, label in TARGETS.items():
    y = y_dict[col]; rf = final[col]['RF']; xgbm = final[col]['XGBoost']
    perm = permutation_importance(rf, X, y, n_repeats=25,
                                   random_state=RAND, scoring='r2')
    for fi, feat in enumerate(feat_names):
        fi_rows.append(dict(target=label, col=col, feature=feat,
                            feat_label=FEAT_LABELS[feat],
                            RF_impurity=round(float(rf.feature_importances_[fi]),5),
                            Perm_mean=round(float(perm.importances_mean[fi]),5),
                            Perm_std=round(float(perm.importances_std[fi]),5)))
    if SHAP:
        feat_df = pd.DataFrame(X, columns=feat_names)
        expl_xgb = shap.TreeExplainer(xgbm)
        sv_xgb   = expl_xgb.shap_values(feat_df); shap_vals_all[col] = sv_xgb
        expl_rf  = shap.TreeExplainer(rf)
        sv_rf    = expl_rf.shap_values(feat_df)
        # sklearn RF regressor TreeExplainer can return a list (one per
        # output) for some SHAP versions; normalize to a single 2D array.
        if isinstance(sv_rf, list):
            sv_rf = sv_rf[0]
        shap_vals_rf_all[col] = sv_rf
        mean_abs_xgb = np.abs(sv_xgb).mean(axis=0)
        mean_abs_rf  = np.abs(sv_rf).mean(axis=0)
        for fn, sv_m, sv_rf_m in zip(feat_names, mean_abs_xgb, mean_abs_rf):
            shap_rows.append(dict(target=label, col=col, feature=fn,
                                  feat_label=FEAT_LABELS[fn],
                                  mean_abs_shap_xgb=round(float(sv_m),5),
                                  mean_abs_shap_rf=round(float(sv_rf_m),5)))
    else:
        perm_sub = (pd.DataFrame(fi_rows)[pd.DataFrame(fi_rows)['col']==col]
                    .sort_values('Perm_mean', ascending=False))
        for _, rw in perm_sub.iterrows():
            shap_rows.append(dict(target=label, col=col,
                                  feature=rw['feature'],
                                  feat_label=rw['feat_label'],
                                  mean_abs_shap_xgb=max(0.0, rw['Perm_mean']),
                                  mean_abs_shap_rf=np.nan))
df_fi   = pd.DataFrame(fi_rows)
df_shap = pd.DataFrame(shap_rows)

if PRIMARY_TARGET in TARGETS and SHAP:
    top3_xgb = (df_shap[df_shap['col']==PRIMARY_TARGET]
                .sort_values('mean_abs_shap_xgb', ascending=False)['feat_label'].head(3).tolist())
    top3_rf  = (df_shap[df_shap['col']==PRIMARY_TARGET]
                .sort_values('mean_abs_shap_rf', ascending=False)['feat_label'].head(3).tolist())
    print(f"    Fmax top-3 SHAP (XGBoost): {top3_xgb}")
    print(f"    Fmax top-3 SHAP (RF, robustness check): {top3_rf}")


# ── Bayesian Optimization (Sec. 4.2) ──────────────────────────────────────────
# Story: given the surrogate, where should the next physical specimen be
# printed? FIXED in three ways relative to v9.0 (see header notes):
#   (1) candidates are generated by sweeping theta in [35, 75] deg through the
#       Masters-Evans AI(theta) map, not by gridding AI directly and capping
#       it at 0.42 -- the latter can never propose the paper's own top
#       candidates (V-beta AI=0.56, V-delta AI=0.62);
#   (2) the t >= 0.40 mm printability floor (Sec. 4.2) is enforced by
#       dropping any candidate whose strut thickness falls below it;
#   (3) the acquisition function scores the XGBoost/GPR convex-ensemble mean
#       (ENSEMBLE_W, inversely proportional to CV RMSE) rather than GPR alone,
#       while GPR's posterior sigma alone still drives the uncertainty term,
#       exactly as Sec. 4.2 specifies ("out-of-grid exploration is driven
#       entirely by the GPR posterior variance").
#   (4) a constant-liar-style greedy diversity filter enforces the geometric
#       separation Delta(theta, t, IL) = (5 deg, 0.10 mm, 0.10 mm) between
#       selected candidates (Sec. 4.2), instead of a plain top-K by EI.
print("  Bayesian optimisation...")

PRINT_FLOOR_T = 0.40   # mm, Sec. 4.2 hardware constraint

def make_candidate_row(theta_deg, il_mm, strut_mm):
    ai_v = ai_of_theta(theta_deg)
    por  = ai_to_porosity_pct(ai_v)
    # Feature order must match FEATURES exactly:
    # [auxetic_index, interlayer_mm, strut_mm, porosity_pct, grading_ratio,
    #  cell_aspect_ratio, epoxy_thickness_mm, l_abs_mm, cell_pitch_x_mm]
    return [ai_v, il_mm, strut_mm, por, 1.0, 1.0, 0.300, 0.0, 0.0]

# Search space, Sec. 4.2: theta in [35,75] deg, IL in [0.30,1.30] mm,
# t in [0.30,1.00] mm, before the printability projection.
N_THETA, N_IL, N_ST = 60, 20, 16
theta_grid = np.linspace(35.0, 75.0, N_THETA)
il_grid    = np.linspace(0.30, 1.30, N_IL)
st_grid    = np.linspace(0.30, 1.00, N_ST)

cands_all = np.array([make_candidate_row(th, il, st)
                       for th in theta_grid for il in il_grid for st in st_grid])
theta_of_cand = np.array([th for th in theta_grid for _ in il_grid for _ in st_grid])
il_of_cand    = np.array([il for _ in theta_grid for il in il_grid for _ in st_grid])
st_of_cand    = np.array([st for _ in theta_grid for _ in il_grid for st in st_grid])

# (2) printability floor
_printable = st_of_cand >= PRINT_FLOOR_T
n_dropped = (~_printable).sum()
cands    = cands_all[_printable]
theta_c  = theta_of_cand[_printable]
il_c     = il_of_cand[_printable]
st_c     = st_of_cand[_printable]
print(f"    Printability floor t>={PRINT_FLOOR_T}mm: kept {len(cands)}/{len(cands_all)} "
      f"candidates ({n_dropped} dropped)")

gpr_m = final[PRIMARY_TARGET]['GPR']
xgb_m = final[PRIMARY_TARGET]['XGBoost']
# GPR Pipeline exposes predict(..., return_std=True) through the wrapped
# GaussianProcessRegressor step; scale inputs with the same fitted scaler.
_gpr_step = gpr_m.named_steps['gpr']; _sc_step = gpr_m.named_steps['sc']
mu_gpr, sig_c = _gpr_step.predict(_sc_step.transform(cands), return_std=True)
mu_xgb = xgb_m.predict(cands)
# (3) convex ensemble mean; GPR sigma alone for uncertainty
mu_c = ENSEMBLE_W.get('XGBoost', 0.5)*mu_xgb + ENSEMBLE_W.get('GPR', 0.5)*mu_gpr

y_best  = y_dict[PRIMARY_TARGET].max()
z_ei    = (mu_c - y_best - 1.0)/(sig_c+1e-9)
EI      = (sig_c+1e-9)*(z_ei*spn.cdf(z_ei)+spn.pdf(z_ei))

# (4) constant-liar-style greedy diversity filter, Sec. 4.2:
# Delta(theta, t, IL) = (5 deg, 0.10 mm, 0.10 mm) between selections.
def select_diverse_top(order_idx, theta_c, il_c, st_c, k=15,
                        d_theta=5.0, d_il=0.10, d_st=0.10):
    chosen = []
    for idx in order_idx:
        ok = True
        for c in chosen:
            if (abs(theta_c[idx]-theta_c[c]) < d_theta and
                abs(il_c[idx]-il_c[c]) < d_il and
                abs(st_c[idx]-st_c[c]) < d_st):
                ok = False; break
        if ok:
            chosen.append(idx)
        if len(chosen) >= k:
            break
    return chosen

ei_order = np.argsort(EI)[::-1]
top_ei = select_diverse_top(ei_order, theta_c, il_c, st_c, k=15)
bo_rows = [dict(rank=r+1,
                theta_deg=round(float(theta_c[fi]),1),
                AI=round(float(cands[fi][0]),4),
                interlayer_mm=round(float(cands[fi][1]),2),
                strut_mm=round(float(cands[fi][2]),2),
                porosity_pct=round(float(cands[fi][3]),1),
                pred_force_N=round(float(mu_c[fi]),1),
                pred_force_N_xgb=round(float(mu_xgb[fi]),1),
                pred_force_N_gpr=round(float(mu_gpr[fi]),1),
                pred_uncertainty=round(float(sig_c[fi]),1),
                EI=round(float(EI[fi]),4))
           for r, fi in enumerate(top_ei)]
df_bo = pd.DataFrame(bo_rows)
opt_strut = cands[top_ei[0], 2]
cands_2d  = np.array([make_candidate_row(th, il, opt_strut)
                       for th in theta_grid for il in il_grid])
mu_2d_xgb = xgb_m.predict(cands_2d)
mu_2d_gpr, sig_2d = _gpr_step.predict(_sc_step.transform(cands_2d), return_std=True)
mu_2d = ENSEMBLE_W.get('XGBoost', 0.5)*mu_2d_xgb + ENSEMBLE_W.get('GPR', 0.5)*mu_2d_gpr
MU_GRID  = mu_2d.reshape(N_THETA, N_IL)
SIG_GRID = sig_2d.reshape(N_THETA, N_IL)
AI_GRID  = np.array([ai_of_theta(th) for th in theta_grid])

exp_best = df[df['configuration']=='Ax_45_0.8_1.0']['max_force_N'].mean() \
           if (df['configuration']=='Ax_45_0.8_1.0').any() else y_best
print(f"  BO optimum: theta={theta_c[top_ei[0]]:.1f}deg (AI={cands[top_ei[0],0]:.4f})  "
      f"IL={cands[top_ei[0],1]:.2f}mm  strut={cands[top_ei[0],2]:.2f}mm  "
      f"-> {mu_c[top_ei[0]]:.1f}+/-{sig_c[top_ei[0]]:.1f}N  "
      f"(best training config={exp_best:.0f}N)")
print(f"  Top-4 diverse candidates (compare against paper's V-alpha/beta/delta/epsilon):")
print(df_bo.head(4)[['rank','theta_deg','AI','interlayer_mm','strut_mm','pred_force_N']].to_string(index=False))


# ── PART E: ML FIGURES (Figs 8-15) ────────────────────────────────────────────
print("\nGenerating ML figures (8-15)...")
cv5 = df_cv[df_cv['cv']=='5-fold-GroupKFold'].copy()

with PdfPages(ML_PDF) as pdf:

    # ── Fig 8: CV R^2 vertical bars ────────────────────────────────
    model_names = list(REG_MODELS.keys())
    x_pos = np.arange(len(model_names))
    fig, axes = plt.subplots(1, len(tgt_list), figsize=(3.5*len(tgt_list), 5.5), sharey=False)
    if len(tgt_list) == 1: axes = [axes]
    for axi, col in enumerate(tgt_list):
        ax = axes[axi]
        sub = (cv5[cv5['col']==col].set_index('model')
               .reindex(model_names).reset_index())
        bar_colors = [MODEL_COLORS.get(m, '#888') for m in model_names]
        bars = ax.bar(x_pos, [r['R2_mean'] for _, r in sub.iterrows()],
                      color=bar_colors, alpha=0.85, edgecolor='black',
                      linewidth=0.8, width=0.6, zorder=3)
        ax.errorbar(x_pos, [r['R2_mean'] for _, r in sub.iterrows()],
                    yerr=[r['R2_std'] for _, r in sub.iterrows()],
                    fmt='none', color='black', linewidth=1.1,
                    capsize=3.5, capthick=1.0, zorder=5)
        for xi, (bar, (_, r)) in enumerate(zip(bars, sub.iterrows())):
            label_y = r['R2_mean']+(r['R2_std'] or 0)+0.018
            ax.text(xi, label_y, f"{r['R2_mean']:.3f}", ha='center',
                    va='bottom', fontsize=8, fontfamily=TNR, color='black')
        ax.axhline(0.9, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8.5)
        ax.set_ylim([-0.2, 1.06])
        polish(ax, ylabel='R² (5-fold GroupKFold CV)' if axi==0 else '',
               title=TARGETS[col] + ('  [paper target]' if col==PRIMARY_TARGET else ''))
    handles8 = [mpatches.Patch(color=MODEL_COLORS[m], ec='black',
                                linewidth=0.7, label=m)
                for m in model_names]
    fig.legend(handles=handles8, loc='lower center', ncol=5,
               frameon=True, edgecolor='black', fancybox=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 8 -- CV R2 bars")

    # ── Fig 9(a,b) paper-style: XGBoost vs RF SHAP attribution robustness
    #    check for Fmax (ADDED — reproduces the explicit comparison Sec. 4.3
    #    describes: "the top-three feature hierarchy... is identical"). ──────
    if SHAP and PRIMARY_TARGET in shap_vals_all:
        sub9 = df_shap[df_shap['col']==PRIMARY_TARGET].copy()
        fig9, axes9 = plt.subplots(1, 2, figsize=(11, 5))
        for axi, (val_col, mdl_lbl) in enumerate([('mean_abs_shap_xgb','XGBoost'),
                                                    ('mean_abs_shap_rf','Random Forest')]):
            ax = axes9[axi]
            s9 = sub9.sort_values(val_col, ascending=True)
            y9 = np.arange(len(s9))
            bar_colors9 = [FEAT_COLORS[i%len(FEAT_COLORS)] for i in range(len(s9))][::-1]
            ax.barh(y9, s9[val_col], height=0.6, color=bar_colors9,
                    edgecolor='black', linewidth=0.8, zorder=3)
            ax.set_yticks(y9); ax.set_yticklabels(s9['feat_label'], fontsize=9.5)
            polish(ax, xlabel='Mean |SHAP value| (N)',
                   title=f'{mdl_lbl} attribution for Fmax')
        fig9.suptitle('Attribution robustness check: top-3 ranking should agree across model classes (Sec. 4.3)',
                       fontsize=9, y=1.02, fontfamily=TNR)
        fig9.tight_layout()
        pdf.savefig(fig9, bbox_inches='tight'); plt.close(fig9)
        print("  Fig 9(a,b) -- SHAP attribution robustness check (XGBoost vs RF, Fmax)")

    # ── Fig 9-code (multi-target XGBoost SHAP importance, extra) ──────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)) if len(tgt_list) > 2 else \
                plt.subplots(1, len(tgt_list), figsize=(6*len(tgt_list), 4.5))
    axes = np.array(axes).flatten()
    for axi, col in enumerate(tgt_list):
        ax = axes[axi]
        sub = df_shap[df_shap['col']==col].sort_values('mean_abs_shap_xgb', ascending=True)
        y_p = np.arange(len(sub))
        bar_colors = [FEAT_COLORS[i%len(FEAT_COLORS)] for i in range(len(sub))][::-1]
        bars = ax.barh(y_p, sub['mean_abs_shap_xgb'], height=0.60,
                       color=bar_colors, edgecolor='black', linewidth=0.8, zorder=3)
        ax.set_yticks(y_p); ax.set_yticklabels(sub['feat_label'], fontsize=9.5)
        polish(ax, xlabel='Mean |SHAP value| (XGBoost)', title=TARGETS[col])
        mx = sub['mean_abs_shap_xgb'].max()
        for bar, val in zip(bars, sub['mean_abs_shap_xgb']):
            ax.text(bar.get_width()+mx*0.02, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='left', fontsize=8, fontfamily=TNR)
    for ax in axes[len(tgt_list):]:
        ax.axis('off')
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 9-code -- SHAP global importance, all targets (extra)")

    # ── Fig 10: SHAP Beeswarm or Permutation Importance ───────────
    if SHAP and PRIMARY_TARGET in shap_vals_all:
        fig = plt.figure(figsize=(9, 5.5))
        feat_df_s = pd.DataFrame(X, columns=[FEAT_LABELS[f] for f in feat_names])
        shap.summary_plot(shap_vals_all[PRIMARY_TARGET], feat_df_s,
                          show=False, plot_size=None,
                          color_bar_label='Feature value (normalised)')
        ax10 = plt.gca()
        ax10.spines['top'].set_visible(False)
        ax10.spines['right'].set_visible(False)
        for sp in ['left','bottom']:
            ax10.spines[sp].set_color('black')
            ax10.spines[sp].set_linewidth(0.9)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    else:
        fig, axes10 = plt.subplots(1, 2, figsize=(11, 5))
        for axi, (col, label) in enumerate([
                (PRIMARY_TARGET, TARGETS[PRIMARY_TARGET]),
                (tgt_list[1] if len(tgt_list) > 1 else PRIMARY_TARGET,
                 TARGETS[tgt_list[1]] if len(tgt_list) > 1 else TARGETS[PRIMARY_TARGET])]):
            ax = axes10[axi]
            sub = df_fi[df_fi['col']==col].sort_values('Perm_mean', ascending=True)
            ax.barh(range(len(sub)), sub['Perm_mean'], xerr=sub['Perm_std'], height=0.55,
                    color=[FEAT_COLORS[i%len(FEAT_COLORS)] for i in range(len(sub))],
                    edgecolor='black', linewidth=0.8,
                    error_kw=dict(ecolor='black', elinewidth=1.0, capsize=3, capthick=1.0))
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['feat_label'], fontsize=9.5)
            polish(ax, xlabel='Permutation Importance (dR2)', title=label)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 10 -- SHAP beeswarm / permutation importance")

    # ── Fig 11-code: SHAP Dependence plots (paper's Fig 9(d) equivalent) ────
    cmap_dep = LinearSegmentedColormap.from_list(
        'dep', [(0,'#0072B2'),(0.5,'#FFFACD'),(1,'#D55E00')])
    if SHAP and PRIMARY_TARGET in shap_vals_all:
        sv_f = shap_vals_all[PRIMARY_TARGET]
        fi_ai = feat_names.index('auxetic_index')
        fi_il = feat_names.index('interlayer_mm')
        fig, axes11 = plt.subplots(1, 2, figsize=(11, 5))
        for ax, (xf, cf, xl, cl) in zip(axes11, [
                (fi_ai, fi_il, 'Auxetic Index (AI)', 'Interlayer (mm)'),
                (fi_il, fi_ai, 'Interlayer (mm)', 'Auxetic Index (AI)')]):
            sc = ax.scatter(X[:,xf], sv_f[:,xf], c=X[:,cf], cmap=cmap_dep,
                            s=32, alpha=0.85, edgecolors='black',
                            linewidths=0.4, zorder=5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
            cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
            cb.set_label(cl, fontsize=9.5, fontfamily=TNR)
            cb.outline.set_edgecolor('black')
            polish(ax, xlabel=xl, ylabel=f'SHAP value for {xl.split(" ")[0]}')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(X[:,feat_names.index('auxetic_index')], y_dict[PRIMARY_TARGET],
                        c=X[:,feat_names.index('interlayer_mm')],
                        cmap=cmap_dep, s=30, alpha=0.85, edgecolors='black', linewidths=0.4)
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
        cb.set_label('Interlayer (mm)', fontsize=9.5, fontfamily=TNR)
        cb.outline.set_edgecolor('black')
        polish(ax, xlabel='Auxetic Index (AI)', ylabel='Peak Force (N)')
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print("  Fig 11-code -- SHAP dependence plots")

    # ── Fig 12-code: Actual vs Predicted (LOGO-CV, extra) ──────────
    def _gp2(c):
        for p in GROUP_ORDER:
            if str(c).startswith(p): return p
        return 'Other'
    grp_labels_all = df['configuration'].apply(_gp2).values
    unique_grps    = [g for g in GROUP_ORDER if g in grp_labels_all]

    n_t = len(tgt_list)
    fig12, axes12 = plt.subplots(2, 2, figsize=(11, 10)) if n_t > 2 else \
                    plt.subplots(1, n_t, figsize=(6*n_t, 5.5))
    axes12 = np.array(axes12).flatten()
    for axi, col in enumerate(tgt_list):
        ax = axes12[axi]; y = y_dict[col]
        m  = build_tuned('XGBoost')
        yp = cross_val_predict(m, X, y, cv=LeaveOneGroupOut(), groups=groups_logo)
        r2v  = r2_score(y, yp); mav  = mean_absolute_error(y, yp)
        rmsv = np.sqrt(mean_squared_error(y, yp))
        for grp in unique_grps:
            mask = (grp_labels_all==grp)
            ax.scatter(y[mask], yp[mask], s=20, alpha=0.85, color=GPAL.get(grp, '#888'),
                       edgecolors='black', linewidths=0.4, zorder=4, label=GLABELS.get(grp, grp))
        lims = [min(y.min(),yp.min())*0.96, max(y.max(),yp.max())*1.04]
        ax.plot(lims, lims, 'k--', linewidth=1.0, alpha=0.8, zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.text(0.05, 0.97, f'R2={r2v:.3f}\nMAE={mav:.1f}\nRMSE={rmsv:.1f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8.5, fontfamily=TNR,
                bbox=dict(fc='white', ec='black', lw=0.8, boxstyle='round,pad=0.3'))
        polish(ax, xlabel=f'Actual {TARGETS[col]}', ylabel=f'Predicted {TARGETS[col]}',
               title=TARGETS[col])
        leg12 = ax.legend(loc='lower right', fontsize=7, ncol=2, frameon=True,
                          edgecolor='black', fancybox=False)
        leg12.get_frame().set_linewidth(0.7)
    for ax in axes12[n_t:]:
        ax.axis('off')
    fig12.tight_layout()
    pdf.savefig(fig12, bbox_inches='tight'); plt.close(fig12)
    print("  Fig 12-code -- Actual vs Predicted (LOGO-CV, extra)")

    # ── Fig 13-code: Confusion Matrix (extra, failure-mode classifier) ─────
    fig13, axes13 = plt.subplots(1, 2, figsize=(11, 5.5))
    cm_cmap = LinearSegmentedColormap.from_list('cm_blue', [(0,'#EBF5FB'),(1,'#0072B2')])
    for axi, (mn, model) in enumerate(CLS_MODELS.items()):
        ax = axes13[axi]
        yp2 = cross_val_predict(model, X, y_cls, cv=skf5)
        cm  = confusion_matrix(y_cls, yp2)
        cm_pct = cm.astype(float)/cm.sum(axis=1, keepdims=True)*100
        ax.imshow(cm_pct, cmap=cm_cmap, vmin=0, vmax=100, aspect='equal')
        for i in range(len(cls_names)):
            for j in range(len(cls_names)):
                v = cm_pct[i, j]
                fc_t = 'white' if v > 50 else 'black'
                ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=10, color=fc_t)
        ax.set_xticks(range(len(cls_names)))
        ax.set_xticklabels([c.replace('_','\n') for c in cls_names], fontsize=9)
        ax.set_yticks(range(len(cls_names))); ax.set_yticklabels(cls_names, fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('black'); sp.set_linewidth(0.9)
        acc_v = accuracy_score(y_cls, yp2); f1w_v = f1_score(y_cls, yp2, average='weighted')
        ax.set_xlabel('Predicted', fontsize=10, fontfamily=TNR)
        ax.set_ylabel('True', fontsize=10, fontfamily=TNR)
        ax.set_title(f'{mn}  Acc={acc_v:.3f}  F1-w={f1w_v:.3f}', fontsize=9.5, pad=6, fontfamily=TNR)
    fig13.tight_layout()
    pdf.savefig(fig13, bbox_inches='tight'); plt.close(fig13)
    print("  Fig 13-code -- Confusion matrix (extra)")

    # ── Fig 14: BO Design Space ────────────────────────────────────
    fig14, axes14 = plt.subplots(1, 2, figsize=(13, 5.5))
    AI_G, IL_G = np.meshgrid(AI_GRID, il_grid, indexing='ij')

    ax = axes14[0]
    cf14 = ax.contourf(IL_G, AI_G, MU_GRID, levels=20, cmap='RdYlGn', zorder=2)
    ax.contour(IL_G, AI_G, MU_GRID, levels=10, colors='black', linewidths=0.35, alpha=0.35)
    cb14 = plt.colorbar(cf14, ax=ax, fraction=0.04, pad=0.04)
    cb14.set_label('Predicted Peak Force (N), XGB/GPR ensemble', fontsize=9.5, fontfamily=TNR)
    cb14.outline.set_edgecolor('black')
    for _, row in df.drop_duplicates('configuration').iterrows():
        ai_v = float(row['auxetic_index'])
        if ai_v > 0:
            ax.scatter(float(row['interlayer_mm']), ai_v, c=[float(row['max_force_N'])],
                       cmap='RdYlGn', vmin=MU_GRID.min(), vmax=MU_GRID.max(),
                       s=22, edgecolors='black', linewidths=0.6, zorder=7)
    opt_il = cands[top_ei[0],1]; opt_ai = cands[top_ei[0],0]; opt_F = mu_c[top_ei[0]]
    ax.scatter(opt_il, opt_ai, marker='*', s=280, color='white',
               edgecolors='black', linewidths=1.4, zorder=9)
    ax.annotate(f'BO optimum\n(AI={opt_ai:.3f}, IL={opt_il:.2f}mm)\nPred~{opt_F:.0f}N',
                xy=(opt_il, opt_ai), xytext=(0.06, 1.07), textcoords='axes fraction',
                fontsize=8.5, fontfamily=TNR,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9),
                bbox=dict(fc='white', ec='black', lw=0.8, boxstyle='round,pad=0.3'))
    ax.set_xlim([0.25, 1.35]); ax.set_ylim([AI_GRID.min()-0.02, AI_GRID.max()+0.02])
    for ta in [0.035, 0.155, 0.414]:
        ax.axhline(ta, color='white', linewidth=0.6, linestyle=':', alpha=0.5)
    polish(ax, xlabel='Interlayer Thickness (mm)', ylabel='Auxetic Index (AI)',
           title=f'Ensemble prediction surface (strut={opt_strut:.2f}mm fixed)')

    ax = axes14[1]
    cmap_u = LinearSegmentedColormap.from_list('unc', [(0,'#EBF5FB'),(0.5,'#6BAED6'),(1,'#1D3557')])
    cf_u = ax.contourf(IL_G, AI_G, SIG_GRID, levels=20, cmap=cmap_u, zorder=2)
    ax.contour(IL_G, AI_G, SIG_GRID, levels=8, colors='black', linewidths=0.3, alpha=0.35)
    cb_u = plt.colorbar(cf_u, ax=ax, fraction=0.04, pad=0.04)
    cb_u.set_label('GPR Predictive Uncertainty sigma (N)', fontsize=9.5, fontfamily=TNR)
    cb_u.outline.set_edgecolor('black')
    for _, row in df.drop_duplicates('configuration').iterrows():
        ai_v = float(row['auxetic_index'])
        if ai_v > 0:
            ax.scatter(float(row['interlayer_mm']), ai_v, color='white',
                       s=20, edgecolors='black', linewidths=0.7, zorder=7)
    top5_cands = cands[top_ei[:5]]
    ax.scatter(top5_cands[:,1], top5_cands[:,0], marker='*', s=180, color='#FFD700',
               edgecolors='black', linewidths=0.9, zorder=9)
    ax.set_xlim([0.25, 1.35]); ax.set_ylim([AI_GRID.min()-0.02, AI_GRID.max()+0.02])
    handles_u = [
        mpatches.Patch(color='#1D3557', alpha=0.8, label='High uncertainty'),
        mpatches.Patch(color='#EBF5FB', alpha=0.8, label='Low uncertainty'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
               markeredgecolor='black', markersize=11, label='Top-5 BO candidates'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Tested configuration'),
    ]
    ax.legend(handles=handles_u, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=True, edgecolor='black', fancybox=False,
              fontsize=8.5).get_frame().set_linewidth(0.8)
    polish(ax, xlabel='Interlayer Thickness (mm)', ylabel='Auxetic Index (AI)',
           title='GPR Predictive Uncertainty')
    fig14.tight_layout(rect=[0, 0.06, 1, 1])
    pdf.savefig(fig14, bbox_inches='tight'); plt.close(fig14)
    print("  Fig 14 -- BO design space (ensemble mean, printability-filtered)")

    # ── Fig 15: BO Candidate Ranking ──────────────────────────────
    fig15, axes15 = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes15[0]
    show_n      = min(10, len(df_bo))
    df_bo_top   = df_bo.head(show_n).iloc[::-1].reset_index(drop=True)
    y_p15       = np.arange(show_n)
    ylabels15   = [f"th={r.theta_deg:.0f} deg / IL={r.interlayer_mm:.2f}mm / St={r.strut_mm:.2f}mm"
                   for _, r in df_bo_top.iterrows()]
    bar_cols15  = plt.cm.RdYlGn(np.linspace(0.4, 0.95, show_n))
    max_pred15  = df_bo_top['pred_force_N'].max(); max_unc15 = df_bo_top['pred_uncertainty'].max()
    bars15      = ax.barh(y_p15, df_bo_top['pred_force_N'], height=0.58,
                          color=bar_cols15, edgecolor='black', linewidth=0.8, zorder=3)
    ax.errorbar(df_bo_top['pred_force_N'], y_p15, xerr=df_bo_top['pred_uncertainty'],
                fmt='none', color='black', linewidth=1.1, capsize=3.5, capthick=1.1, zorder=5)
    ax.axvline(exp_best, color='#D55E00', linewidth=1.4, linestyle='--', zorder=2)
    for bar, (_, r) in zip(bars15, df_bo_top.iterrows()):
        val_x = r['pred_force_N']+r['pred_uncertainty']+7
        ax.text(val_x, bar.get_y()+bar.get_height()/2, f"{r['pred_force_N']:.0f}",
                va='center', ha='left', fontsize=8.5, fontfamily=TNR)
    ax.set_yticks(y_p15); ax.set_yticklabels(ylabels15, fontsize=8)
    leg15a = [Line2D([0],[0], color='#D55E00', linewidth=1.4, linestyle='--',
                     label=f'Best training config: {exp_best:.0f} N')]
    ax.legend(handles=leg15a, loc='lower center', bbox_to_anchor=(0.5, -0.16), frameon=True,
              edgecolor='black', fancybox=False, fontsize=8.5).get_frame().set_linewidth(0.8)
    x_max15 = max_pred15+max_unc15+65
    ax.set_xlim([min(df_bo_top['pred_force_N'].min()-max_unc15-65, exp_best-110), x_max15])
    polish(ax, xlabel='Predicted Peak Force (N)', title=f'Top-{show_n} BO Candidates (diversity-filtered)')

    ax = axes15[1]
    ranks15     = df_bo['rank'].values[:15]
    pred_F_vals = df_bo['pred_force_N'].values[:15]
    ei_vals     = df_bo['EI'].values[:15]
    ax2_15      = ax.twinx()
    l1, = ax.plot(ranks15, pred_F_vals, color='#009E73', linewidth=1.8, marker='o', markersize=6,
                  markeredgecolor='black', markeredgewidth=0.7, zorder=5, label='Predicted Force (N)')
    l2, = ax2_15.plot(ranks15, ei_vals, color='#D55E00', linewidth=1.4, marker='s', markersize=5,
                      markeredgecolor='black', markeredgewidth=0.6, linestyle='--', zorder=4,
                      label='Expected Improvement')
    ax.axhline(exp_best, color='#555', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax2_15.spines['top'].set_visible(False); ax2_15.spines['left'].set_visible(False)
    ax2_15.spines['bottom'].set_visible(False)
    ax2_15.spines['right'].set_color('#D55E00'); ax2_15.spines['right'].set_linewidth(0.9)
    ax2_15.tick_params(axis='y', colors='#D55E00', width=0.9, direction='in')
    ax2_15.set_ylabel('Expected Improvement', fontsize=9.5, color='#D55E00',
                      fontfamily=TNR, rotation=90, labelpad=6)
    polish(ax, xlabel='Rank (sorted by EI, diversity-filtered)', ylabel='Predicted Peak Force (N)',
           title='BO Candidate Ranking')
    lines15 = [l1, l2, Line2D([0],[0], color='#555', lw=1.0, ls=':',
                       label=f'Best training config ({exp_best:.0f} N)')]
    ax.legend(handles=lines15, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=1,
              frameon=True, edgecolor='black', fancybox=False, fontsize=8.5).get_frame().set_linewidth(0.8)
    fig15.tight_layout(rect=[0, 0.07, 1, 0.96])
    pdf.savefig(fig15, bbox_inches='tight'); plt.close(fig15)
    print("  Fig 15 -- BO candidate ranking")

print(f"  ML figures -> {ML_PDF}")


# ── PART G: DESIGN SPACE FIGURES (Figs 16-18) ─────────────────────────────────
# Story: zooming out from the discrete training grid to the continuous
# geometry space -- where do the theoretical optimum and the print-floor
# boundary actually sit relative to what was tested?
print("\nGenerating design space figures (16-18)...")

# Quadratic trends through 45/60/75 deg means per IL level
fits_F_ds, fits_E_ds = {}, {}
for il in [1.0, 0.8, 0.6]:
    ai_pts, F_pts, E_pts = [], [], []
    for ang in [45, 60, 75]:
        sub = df[df['configuration'].str.contains(f'Ax_{ang}') &
                 (abs(df['interlayer_mm']-il) < 0.01)]
        if len(sub):
            ai_pts.append(ai_of_theta(ang))
            F_pts.append(sub['max_force_N'].mean())
            E_pts.append(sub['energy_Nmm'].mean())
    if len(ai_pts) >= 3:
        fits_F_ds[il] = np.polyfit(ai_pts, F_pts, 2)
        fits_E_ds[il] = np.polyfit(ai_pts, E_pts, 2)

# Dedicated GPR and RF for design-space visualization (tuned architecture)
_sc_ds  = StandardScaler(); _Xs_ds = _sc_ds.fit_transform(X)
_gpr_ds = build_tuned('GPR').named_steps['gpr']
_gpr_ds.fit(_Xs_ds, y_dict[PRIMARY_TARGET])
_rf_ds  = build_tuned('RandomForest')
_rf_ds.fit(X, y_dict[PRIMARY_TARGET])

ang_range_ds = np.linspace(28, 82, 300)
ai_range_ds  = np.array([ai_of_theta(a) for a in ang_range_ds])

with PdfPages(DESIGN_PDF) as pdf_ds:

    # ── Fig 16: Polynomial Trend Extrapolation ────────────────────
    fig16, axes16 = plt.subplots(1, 2, figsize=(13, 5.5))
    for axi, (col, y_lbl, fits_dict) in enumerate([
            ('max_force_N', 'Peak Force (N)',         fits_F_ds),
            ('energy_Nmm',  'Fracture Energy (N·mm)', fits_E_ds)]):
        ax = axes16[axi]
        for il in [1.0, 0.8, 0.6]:
            c = fits_dict.get(il)
            if c is None: continue
            curve = np.polyval(c, ai_range_ds)
            ax.plot(ang_range_ds, curve, color=IL_COLORS_DS[il], linewidth=2.0,
                    label=f'IL = {il} mm', zorder=5)
            ai_star = -c[1]/(2*c[0])
            if 0.0 < ai_star < 1.5:
                ang_star = theta_of_ai(ai_star)
                if not np.isnan(ang_star) and 28 < ang_star < 82:
                    F_star = np.polyval(c, ai_star)
                    ax.plot(ang_star, F_star, marker='*', markersize=13,
                            color=IL_COLORS_DS[il], markeredgecolor='black',
                            markeredgewidth=0.9, zorder=8, linestyle='none')
                    ax.annotate(f'theta*~{ang_star:.0f} deg', (ang_star, F_star),
                                xytext=(6, 5), textcoords='offset points',
                                fontsize=8.5, fontfamily=TNR, color=IL_COLORS_DS[il],
                                bbox=dict(fc='white', ec='none', alpha=0.85))
        for il in [1.0, 0.8, 0.6]:
            for ang in [45, 60, 75]:
                sub = df[df['configuration'].str.contains(f'Ax_{ang}') &
                         (abs(df['interlayer_mm']-il) < 0.01)]
                if len(sub):
                    ax.scatter(ang, sub[col].mean(), color=IL_COLORS_DS[il], s=55,
                               edgecolors='black', linewidths=1.0, zorder=7)
        ax.axvspan(28, 44, alpha=0.07, color='#D55E00', zorder=1)
        ax.axvspan(44, 82, alpha=0.04, color='#009E73', zorder=1)
        ax.axvline(45, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ylims = ax.get_ylim()
        ax.text(36, ylims[0]+(ylims[1]-ylims[0])*0.05, 'Extrapolation\nregion',
                ha='center', fontsize=8.5, fontfamily=TNR, color='#D55E00', style='italic')
        ax.text(63, ylims[0]+(ylims[1]-ylims[0])*0.05, 'Tested region',
                ha='center', fontsize=8.5, fontfamily=TNR, color='#009E73', style='italic')
        ax.set_xlim([28, 82]); ax.invert_xaxis()
        ax.set_xticks([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        polish(ax, xlabel='Cell Angle theta (deg)', ylabel=y_lbl)
        handles16 = [Line2D([0],[0], color=c16, linewidth=2.0, label=f'IL = {k16} mm')
                     for k16, c16 in IL_COLORS_DS.items()] + [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#555',
                   markeredgecolor='black', markersize=8, label='Experimental mean'),
            Line2D([0],[0], marker='*', color='w', markerfacecolor='#555',
                   markeredgecolor='black', markersize=12, label='Predicted optimum theta*'),
        ]
        ax.legend(handles=handles16, loc='upper right', frameon=True,
                  edgecolor='black', fancybox=False, fontsize=8.5)
    fig16.tight_layout()
    pdf_ds.savefig(fig16, bbox_inches='tight'); plt.close(fig16)
    print("  Fig 16 -- Polynomial trend extrapolation")

    # ── Fig 17: RF 2-D Surface + GPR Smooth Angle Sensitivity ─────
    ang_2d = np.linspace(30, 80, 100)
    il_2d  = np.linspace(0.30, 1.35, 80)
    ANG2D, IL2D = np.meshgrid(ang_2d, il_2d, indexing='ij')
    F_surf = np.zeros_like(ANG2D)
    for ai_i, ang_v in enumerate(ang_2d):
        for il_i, il_v in enumerate(il_2d):
            ai_v = ai_of_theta(ang_v); po = ai_to_porosity_pct(ai_v)
            F_surf[ai_i, il_i] = _rf_ds.predict(
                [[ai_v, il_v, 0.8, po, 1.0, 1.0, 0.300, 0.0, 0.0]])[0]

    cmap_rg = LinearSegmentedColormap.from_list('rg', [(0,'#0072B2'),(0.45,'#FFFACD'),(1,'#D55E00')])
    fig17, axes17 = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes17[0]
    cf17 = ax.contourf(IL2D, ANG2D, F_surf, levels=20, cmap=cmap_rg)
    ax.contour(IL2D, ANG2D, F_surf, levels=10, colors='black', linewidths=0.35, alpha=0.35)
    cb17 = plt.colorbar(cf17, ax=ax, fraction=0.04, pad=0.04)
    cb17.set_label('Predicted Peak Force (N)', fontsize=9.5, fontfamily=TNR)
    cb17.outline.set_edgecolor('black')
    for _, row in df.drop_duplicates('configuration').iterrows():
        ai_v = float(row['auxetic_index'])
        if ai_v <= 0: continue
        ang_r = theta_of_ai(ai_v)
        if np.isnan(ang_r): continue
        ax.scatter(float(row['interlayer_mm']), ang_r, c=[float(row['max_force_N'])],
                   cmap=cmap_rg, vmin=F_surf.min(), vmax=F_surf.max(),
                   s=22, edgecolors='black', linewidths=0.6, zorder=7)
    opt_idx17    = np.unravel_index(F_surf.argmax(), F_surf.shape)
    opt_ang_ds   = ang_2d[opt_idx17[0]]; opt_il_ds = il_2d[opt_idx17[1]]; opt_F_ds = F_surf[opt_idx17]
    ax.scatter(opt_il_ds, opt_ang_ds, marker='*', s=280, color='white',
               edgecolors='black', linewidths=1.4, zorder=9)
    ax.annotate(f'RF optimum\ntheta={opt_ang_ds:.0f}deg, IL={opt_il_ds:.2f}mm\nPred~{opt_F_ds:.0f}N',
                (opt_il_ds, opt_ang_ds), xytext=(8, 6), textcoords='offset points',
                fontsize=8.5, fontfamily=TNR,
                bbox=dict(fc='white', ec='black', lw=0.8, boxstyle='round,pad=0.3'))
    for ta in [45, 60, 75]:
        ax.axhline(ta, color='white', linewidth=0.6, linestyle=':', alpha=0.55)
    ax.set_yticks([35, 40, 45, 50, 55, 60, 65, 70, 75])
    ax.set_xlim([0.25, 1.35]); ax.set_ylim([33, 80])
    polish(ax, xlabel='Interlayer Thickness (mm)', ylabel='Cell Angle theta (deg)',
           title='RF Prediction Surface (strut=0.8 mm)')

    ax = axes17[1]
    angs_1d = np.linspace(30, 80, 150)
    for il_v, ilc, ils in [(0.6, IL_COLORS_DS[0.6], ':'),
                            (0.8, IL_COLORS_DS[0.8], '--'),
                            (1.0, IL_COLORS_DS[1.0], '-')]:
        X_1d = np.array([[ai_of_theta(a), il_v, 0.8, ai_to_porosity_pct(ai_of_theta(a)),
                   1.0, 1.0, 0.300, 0.0, 0.0] for a in angs_1d])
        X_1d_sc  = _sc_ds.transform(X_1d)
        F_mu, F_sig = _gpr_ds.predict(X_1d_sc, return_std=True)
        ax.plot(angs_1d, F_mu, color=ilc, linewidth=1.8, linestyle=ils,
                label=f'IL = {il_v} mm', zorder=5)
        ax.fill_between(angs_1d, F_mu-F_sig, F_mu+F_sig, color=ilc, alpha=0.12, zorder=3)
    for ta in [45, 60, 75]:
        ax.axvline(ta, color='#888888', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvspan(30, 44.5, alpha=0.07, color='#D55E00')
    ax.text(37.5, ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 'Extrapolated',
            ha='center', fontsize=8.5, fontfamily=TNR, color='#D55E00', style='italic')
    ax.set_xlim([28, 82]); ax.invert_xaxis()
    ax.set_xticks([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    polish(ax, xlabel='Cell Angle theta (deg)', ylabel='Predicted Peak Force (N)',
           title='GPR Sensitivity: Force vs Angle (strut=0.8 mm)')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, fontsize=8.5)
    fig17.tight_layout()
    pdf_ds.savefig(fig17, bbox_inches='tight'); plt.close(fig17)
    print("  Fig 17 -- RF surface + GPR smooth angle sensitivity")

    # ── Fig 18: GPR Uncertainty Map + Recommended Next Experiments ──
    ang_u = np.linspace(30, 80, 60)
    il_u  = np.linspace(0.30, 1.30, 45)
    ANG_U, IL_U = np.meshgrid(ang_u, il_u, indexing='ij')
    MU_U  = np.zeros_like(ANG_U); SIG_U = np.zeros_like(ANG_U)
    for ai_i, ang_v in enumerate(ang_u):
        for il_i, il_v in enumerate(il_u):
            ai_v = ai_of_theta(ang_v); po = ai_to_porosity_pct(ai_v)
            x_new = _sc_ds.transform([[ai_v, il_v, 0.8, po, 1.0, 1.0, 0.300, 0.0, 0.0]])
            mu_v, sg_v = _gpr_ds.predict(x_new, return_std=True)
            MU_U[ai_i, il_i] = mu_v[0]; SIG_U[ai_i, il_i] = sg_v[0]

    fig18, ax = plt.subplots(figsize=(10, 6))
    cmap_u2 = LinearSegmentedColormap.from_list('unc2', [(0,'#EBF5FB'),(0.4,'#6BAED6'),(1,'#1D3557')])
    cf_u2 = ax.contourf(IL_U, ANG_U, SIG_U, levels=20, cmap=cmap_u2)
    ax.contour(IL_U, ANG_U, SIG_U, levels=8, colors='black', linewidths=0.3, alpha=0.35)
    cb_u2 = plt.colorbar(cf_u2, ax=ax, fraction=0.04, pad=0.04)
    cb_u2.set_label('GPR Predictive Uncertainty sigma (N)', fontsize=9.5, fontfamily=TNR)
    cb_u2.outline.set_edgecolor('black')
    for _, row in df.drop_duplicates('configuration').iterrows():
        ai_v = float(row['auxetic_index'])
        if ai_v <= 0: continue
        ang_r = theta_of_ai(ai_v)
        if np.isnan(ang_r): continue
        ax.scatter(float(row['interlayer_mm']), ang_r, color='white', s=28,
                   edgecolors='black', linewidths=0.9, zorder=7)
    y_best_ds = y_dict[PRIMARY_TARGET].max()
    next_exps = []
    for ai_i, ang_v in enumerate(ang_u):
        for il_i, il_v in enumerate(il_u):
            mu_v = MU_U[ai_i, il_i]; sg_v = SIG_U[ai_i, il_i]
            z_ei = (mu_v-y_best_ds-0.5)/(sg_v+1e-9)
            ei   = (sg_v+1e-9)*(spn.cdf(z_ei)*z_ei+spn.pdf(z_ei))
            next_exps.append((ang_v, il_v, mu_v, sg_v, ei))
    next_exps.sort(key=lambda x: -x[4]); top5_ds = next_exps[:5]
    offsets_ds = [(10,-8),(10,6),(-60,-8),(-60,6),(10,-20)]
    for rank, (ang_v, il_v, mu_v, sg_v, _ei) in enumerate(top5_ds, 1):
        ax.scatter(il_v, ang_v, marker='*', s=220, color='#FFD700',
                   edgecolors='black', linewidths=0.9, zorder=9)
        dx, dy = offsets_ds[rank-1]
        ax.annotate(f'#{rank}: theta~{ang_v:.0f}deg\nIL={il_v:.2f}mm\nmu~{mu_v:.0f}N',
                    (il_v, ang_v), xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.5, fontfamily=TNR,
                    bbox=dict(fc='white', ec='#888', lw=0.7, alpha=0.92, boxstyle='round,pad=0.25'),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=0.7))
    for ta in [45, 60, 75]:
        ax.axhline(ta, color='white', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.text(1.26, ta, f'{ta} deg', va='center', fontsize=8.5, fontfamily=TNR, color='white')
    ax.set_xlim([0.25, 1.35]); ax.set_ylim([28, 82])
    ax.set_yticks([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    polish(ax, xlabel='Interlayer Thickness (mm)', ylabel='Cell Angle theta (deg)')
    handles_u3 = [
        mpatches.Patch(color='#1D3557', alpha=0.85, label='High uncertainty -> prioritise'),
        mpatches.Patch(color='#EBF5FB', alpha=0.85, label='Low uncertainty -> well-covered'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
               markeredgecolor='black', markersize=12, label='Recommended next experiment'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Already tested'),
    ]
    ax.legend(handles=handles_u3, loc='lower right', frameon=True, edgecolor='black',
              fancybox=False, fontsize=8.5).get_frame().set_linewidth(0.8)
    fig18.tight_layout()
    pdf_ds.savefig(fig18, bbox_inches='tight'); plt.close(fig18)
    print("  Fig 18 -- GPR uncertainty map + next experiments")

print(f"  Design space figures -> {DESIGN_PDF}")
print("\n  Key design space findings:")
print(f"  RF surface optimum: theta~{opt_ang_ds:.0f}deg, IL={opt_il_ds:.2f}mm -> ~{opt_F_ds:.0f} N")


# ── PART I: PHYSICAL VALIDATION -- SURROGATE OPTIMISM GAP (Sec. 4.5 / 5.1) ────
# ADDED: this entire Part is new. v9.0 trained and optimized against the
# 160-specimen training campaign only and never closed the loop -- there was
# no code anywhere that touched the 16 held-out physical-validation specimens
# (V-alpha/beta/delta/epsilon, Table 5) or reproduced the paper's central
# closed-loop claim: that the four BO-recommended, out-of-grid geometries were
# actually printed, tested, and checked against what the surrogate predicted.
# This Part does that: it re-uses the *already-fitted* final[...] ensemble
# from Part D (no re-fitting, no leakage -- these models never saw df_val),
# scores the four validation geometries, and reproduces Table 5 / Fig. 11 /
# Fig. 12(a) from the manuscript.
print("\nPhysical validation -- surrogate optimism gap (Sec. 4.5 / 5.1)...")

if len(df_val) == 0:
    print("  ! No validation specimens matched (df_val is empty) -- skipping Part I.")
    print("    This almost certainly means the 'configuration' column in your real")
    print("    spreadsheet spells the four candidates differently than the regex")
    print("    above expects. Check df['configuration'].unique() and, if needed,")
    print("    adjust the _val_mask pattern near the top of the script, then re-run.")
else:
    df_val[FEATURES] = df_val[FEATURES].fillna(0.0)
    X_val = df_val[FEATURES].values

    # Same ensemble convention as the BO search in Part D: XGBoost + GPR,
    # convex-weighted by ENSEMBLE_W (inverse-CV-RMSE), GPR posterior sigma
    # alone carries the uncertainty (Table 5's "Pred. Fmax +/- ..." column).
    _gpr_val_m = final[PRIMARY_TARGET]['GPR']
    _xgb_val_m = final[PRIMARY_TARGET]['XGBoost']
    _gpr_val_step = _gpr_val_m.named_steps['gpr']; _sc_val_step = _gpr_val_m.named_steps['sc']
    mu_gpr_val, sig_val = _gpr_val_step.predict(_sc_val_step.transform(X_val), return_std=True)
    mu_xgb_val = _xgb_val_m.predict(X_val)
    mu_val = ENSEMBLE_W.get('XGBoost', 0.5)*mu_xgb_val + ENSEMBLE_W.get('GPR', 0.5)*mu_gpr_val
    df_val = df_val.assign(pred_force_N=mu_val, pred_sigma_N=sig_val)

    # Predicted-calculated Etri (Sec. 5.1 / Fig. 11b): "Etri derived from
    # predicted Fmax x measured delta_peak" -- the surrogate only ever predicts
    # Fmax (Sec. 4.1), so the paper's predicted-side energy figure is a hybrid
    # quantity that borrows the measured displacement. Matches the triangular
    # (0.5*F*d) approximation used for energy_Nmm throughout this dataset.
    if 'displacement_mm' in df_val.columns:
        df_val = df_val.assign(pred_etri_Nmm=0.5*df_val['pred_force_N']*df_val['displacement_mm'])

    VAL_ORDER = [v for v in ['V_alpha', 'V_beta', 'V_delta', 'V_epsilon']
                 if v in df_val['val_id'].unique()]

    def _mode_summary(sub):
        vc = sub['failure_mode'].value_counts()
        return ' / '.join(f"{m} ({n})" for m, n in vc.items())

    val_rows = []
    for vid in VAL_ORDER:
        sub = df_val[df_val['val_id'] == vid]
        pred_F = float(sub['pred_force_N'].mean()); sig_F = float(sub['pred_sigma_N'].mean())
        meas_F = float(sub['max_force_N'].mean());  meas_F_sd = float(sub['max_force_N'].std(ddof=1))
        gap_pct = 100.0*(meas_F - pred_F)/pred_F if pred_F else np.nan
        ai_v = float(sub['auxetic_index'].mean())
        theta_v = theta_of_ai(ai_v) if ai_v > 0 else np.nan
        row = dict(val_id=vid, label=VAL_LABELS.get(vid, vid), n=len(sub),
                   theta_deg=round(theta_v, 1) if pd.notna(theta_v) else np.nan,
                   strut_mm=round(float(sub['strut_mm'].mean()), 2),
                   interlayer_mm=round(float(sub['interlayer_mm'].mean()), 2),
                   AI=round(ai_v, 3),
                   pred_force_N=round(pred_F, 1), pred_sigma_N=round(sig_F, 1),
                   meas_force_N_mean=round(meas_F, 1), meas_force_N_sd=round(meas_F_sd, 1),
                   optimism_gap_pct=round(gap_pct, 2),
                   dominant_failure_modes=_mode_summary(sub))
        if 'pred_etri_Nmm' in sub.columns:
            row['pred_etri_Nmm']  = round(float(sub['pred_etri_Nmm'].mean()), 1)
        if 'energy_Nmm' in sub.columns:
            row['meas_etri_Nmm']  = round(float(sub['energy_Nmm'].mean()), 1)
        val_rows.append(row)
    df_validation = pd.DataFrame(val_rows)

    # Training-campaign optimum, for the "did the closed-loop design beat the
    # best thing already in the training grid" comparison (Sec. 5.1).
    _cfg_means = df.groupby('configuration')['max_force_N'].mean()
    best_train_cfg = _cfg_means.idxmax(); best_train_F = float(_cfg_means.max())
    base_F_ref = float(df[df['group'] == 'Base']['max_force_N'].mean()) if 'Base' in df['group'].values else np.nan

    print(f"  Best training configuration: {best_train_cfg} = {best_train_F:.1f} N")
    for _, r in df_validation.iterrows():
        beat = "beats" if r['meas_force_N_mean'] > best_train_F else "below"
        print(f"    {r['label']}: theta~{r['theta_deg']}deg AI={r['AI']}  "
              f"pred={r['pred_force_N']:.0f}+/-{r['pred_sigma_N']:.0f}N  "
              f"meas={r['meas_force_N_mean']:.1f}+/-{r['meas_force_N_sd']:.1f}N  "
              f"gap={r['optimism_gap_pct']:+.2f}%  ({beat} training optimum)")
    _mean_signed = df_validation['optimism_gap_pct'].mean()
    _mean_abs    = df_validation['optimism_gap_pct'].abs().mean()
    print(f"  Mean signed optimism gap: {_mean_signed:+.2f}%  |  mean absolute gap: {_mean_abs:.2f}%")
    if pd.notna(base_F_ref):
        for _, r in df_validation.iterrows():
            pct_over_base = 100.0*(r['meas_force_N_mean']-base_F_ref)/base_F_ref
            print(f"    {r['label']} vs Base: {pct_over_base:+.1f}%")

    # ── Fig. 11: Validation Campaign (predicted vs measured, Etri, failure modes) ─
    print("Generating physical-validation figures (Figs 11-12)...")
    with PdfPages(VALIDATION_PDF) as pdf_v:

        fig11, axes11 = plt.subplots(1, 3, figsize=(17, 5.2))

        # (a) predicted (open, +/-GPR sigma) vs measured (filled, +/-SD) Fmax
        ax = axes11[0]
        xpos = np.arange(len(df_validation))
        w = 0.36
        ax.bar(xpos - w/2, df_validation['pred_force_N'], width=w,
               yerr=df_validation['pred_sigma_N'], capsize=4,
               facecolor='white', edgecolor='black', hatch='///', linewidth=1.1,
               label='Predicted (ensemble +/-1 GPR sigma)', zorder=3)
        ax.bar(xpos + w/2, df_validation['meas_force_N_mean'], width=w,
               yerr=df_validation['meas_force_N_sd'], capsize=4,
               color=[VAL_COLORS.get(v, '#888888') for v in df_validation['val_id']],
               edgecolor='black', linewidth=1.1, alpha=0.9,
               label='Measured (+/-1 SD, n=4)', zorder=3)
        ax.axhline(best_train_F, color='#444444', linestyle='--', linewidth=1.1,
                   label=f'Best training config ({best_train_F:.0f} N)', zorder=2)
        for xi, (_, r) in zip(xpos, df_validation.iterrows()):
            ytop = max(r['pred_force_N']+r['pred_sigma_N'], r['meas_force_N_mean']+r['meas_force_N_sd'])
            ax.text(xi, ytop*1.03, f"{r['optimism_gap_pct']:+.1f}%", ha='center',
                    fontsize=9, fontweight='bold', fontfamily=TNR)
        ax.set_xticks(xpos); ax.set_xticklabels(df_validation['label'], fontsize=11)
        polish(ax, ylabel='Peak Force, Fmax (N)')
        ax.legend(fontsize=7.5, frameon=True, edgecolor='black', fancybox=False, loc='lower right')
        ax.set_title('(a) Predicted vs. measured Fmax', fontsize=10, fontfamily=TNR)

        # (b) predicted-calculated vs measured-calculated Etri
        ax = axes11[1]
        if 'pred_etri_Nmm' in df_validation.columns and 'meas_etri_Nmm' in df_validation.columns:
            ax.bar(xpos - w/2, df_validation['pred_etri_Nmm'], width=w,
                   facecolor='white', edgecolor='black', hatch='///', linewidth=1.1,
                   label='Pred. Fmax x measured delta_peak', zorder=3)
            ax.bar(xpos + w/2, df_validation['meas_etri_Nmm'], width=w,
                   color=[VAL_COLORS.get(v, '#888888') for v in df_validation['val_id']],
                   edgecolor='black', linewidth=1.1, alpha=0.9,
                   label='Measured Fmax x measured delta_peak', zorder=3)
            ax.set_xticks(xpos); ax.set_xticklabels(df_validation['label'], fontsize=11)
            polish(ax, ylabel='Triangular Fracture Energy, Etri (N*mm)')
            ax.legend(fontsize=7.5, frameon=True, edgecolor='black', fancybox=False, loc='lower right')
        else:
            ax.text(0.5, 0.5, 'displacement_mm / energy_Nmm\nnot available in this dataset',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('(b) Predicted- vs. measured-calculated Etri', fontsize=10, fontfamily=TNR)

        # (c) failure-mode distribution, 16 validation specimens (4 bars)
        ax = axes11[2]
        ct_val = pd.crosstab(df_val['val_id'], df_val['failure_mode']).reindex(VAL_ORDER).fillna(0)
        ct_val_pct = ct_val.div(ct_val.sum(axis=1), axis=0)*100
        bottoms_v = np.zeros(len(VAL_ORDER))
        for mode in ct_val_pct.columns:
            vals = ct_val_pct[mode].values
            mkey = str(mode).lower().replace(' ', '_').replace('-', '_')
            ax.bar(range(len(VAL_ORDER)), vals, bottom=bottoms_v,
                   color=MODE_COLORS.get(mkey, '#AAAAAA'), edgecolor='black',
                   linewidth=0.8, alpha=0.88, width=0.6,
                   label=str(mode).replace('_', ' ').title())
            bottoms_v += vals
        ax.set_xticks(range(len(VAL_ORDER)))
        ax.set_xticklabels([VAL_LABELS.get(v, v) for v in VAL_ORDER], fontsize=11)
        ax.set_ylim([0, 112]); ax.set_yticks([0, 25, 50, 75, 100])
        polish(ax, ylabel='Proportion (%)')
        h11c, l11c = ax.get_legend_handles_labels()
        ax.legend(h11c, l11c, fontsize=6.8, frameon=True, edgecolor='black',
                  fancybox=False, loc='upper center', ncol=1, bbox_to_anchor=(1.32, 1.02))
        ax.set_title('(c) Failure-mode distribution (validation)', fontsize=10, fontfamily=TNR)

        fig11.suptitle('Fig. 11 -- Validation campaign: V-alpha, V-beta, V-delta, V-epsilon',
                        fontsize=11, fontfamily=TNR)
        fig11.tight_layout(rect=[0, 0, 0.99, 0.95])
        pdf_v.savefig(fig11, bbox_inches='tight'); plt.close(fig11)
        print("  Fig 11 -- Validation campaign (predicted vs measured, Etri, failure modes)")

        # ── Fig. 12(a): full 176-specimen Fmax vs AI, validation as triangles ──
        # (Fig. 12(b) in the paper is a macro-photograph of the physical V-delta
        # specimen cross-section -- not something a code artifact can reproduce;
        # omitted here, panel (a) only.)
        fig12a, ax = plt.subplots(figsize=(9, 6))
        for grp in GROUP_ORDER:
            sub = df[df['group'] == grp]
            if len(sub):
                ax.scatter(sub['auxetic_index'], sub['max_force_N'],
                           color=GPAL.get(grp, '#888888'), s=42, alpha=0.75,
                           edgecolors='black', linewidths=0.4, label=GLABELS.get(grp, grp), zorder=3)
        for vid in VAL_ORDER:
            sub = df_val[df_val['val_id'] == vid]
            ax.scatter(sub['auxetic_index'], sub['max_force_N'], marker='^', s=110,
                       color=VAL_COLORS.get(vid, '#000000'), edgecolors='black',
                       linewidths=0.9, label=VAL_LABELS.get(vid, vid), zorder=6)
        ax.axhline(best_train_F, color='#444444', linestyle='--', linewidth=1.0, zorder=2,
                   label=f'Best training config ({best_train_F:.0f} N)')
        polish(ax, xlabel='Auxetic Index, AI', ylabel='Peak Force, Fmax (N)')
        ax.legend(fontsize=7, frameon=True, edgecolor='black', fancybox=False,
                  ncol=2, loc='lower right')
        ax.set_title('Fig. 12(a) -- Full 176-specimen Fmax vs. AI '
                      '(training = circles, validation = triangles)', fontsize=10, fontfamily=TNR)
        fig12a.tight_layout()
        pdf_v.savefig(fig12a, bbox_inches='tight'); plt.close(fig12a)
        print("  Fig 12(a) -- Full-cohort Fmax vs AI scatter (validation highlighted)")

        # ── Fig 11(d)-code (extra): training-cohort vs validation-cohort failure- ─
        # mode shift, quantifying the qualitative claim in Sec. 5.1's closing
        # paragraph (45deg training: Mixed/Cohesive-Mixed-dominated vs. the
        # validation cohort's Cohesive-Interlayer dominance). Not itself one of
        # the paper's numbered figures -- kept as a direct visual check of that
        # claim, clearly labeled as an extra.
        fig11d, ax = plt.subplots(figsize=(7, 5.2))
        ct_45 = df[df['group'] == 'Ax_45']['failure_mode'].value_counts(normalize=True)*100
        ct_v  = df_val['failure_mode'].value_counts(normalize=True)*100
        cohorts = pd.DataFrame({'45deg training cohort': ct_45,
                                 'All 16 validation specimens': ct_v}).fillna(0).T
        bottoms_d = np.zeros(len(cohorts))
        for mode in cohorts.columns:
            vals = cohorts[mode].values
            mkey = str(mode).lower().replace(' ', '_').replace('-', '_')
            ax.barh(range(len(cohorts)), vals, left=bottoms_d,
                    color=MODE_COLORS.get(mkey, '#AAAAAA'), edgecolor='black',
                    linewidth=0.8, alpha=0.88, height=0.55,
                    label=str(mode).replace('_', ' ').title())
            bottoms_d += vals
        ax.set_yticks(range(len(cohorts))); ax.set_yticklabels(cohorts.index, fontsize=9.5)
        ax.set_xlim([0, 100])
        polish(ax, xlabel='Proportion (%)')
        h11d, l11d = ax.get_legend_handles_labels()
        ax.legend(h11d, l11d, fontsize=7, frameon=True, edgecolor='black',
                  fancybox=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.12))
        ax.set_title('Fig 11(d)-code -- Training vs. validation failure-mode shift (extra)',
                      fontsize=9.5, fontfamily=TNR)
        fig11d.tight_layout(rect=[0, 0.1, 1, 1])
        pdf_v.savefig(fig11d, bbox_inches='tight'); plt.close(fig11d)
        print("  Fig 11(d)-code -- Training-vs-validation failure-mode shift (extra)")

    print(f"  Validation figures -> {VALIDATION_PDF}")


# ── PART F: EXCEL EXPORT ──────────────────────────────────────────────────────
print("\nExporting Excel tables...")

def _tb(s=Side(style='thin', color='CCCCCC')):
    return Border(left=s, right=s, top=s, bottom=s)

def write_df(ws, df_data, hf='1F3864', hc='FFFFFF'):
    fill  = PatternFill('solid', fgColor=hf)
    hfnt  = Font(bold=True, color=hc, size=10, name=TNR)
    hal   = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.append(list(df_data.columns)); ws.row_dimensions[1].height = 36
    for cell in ws[1]:
        cell.fill = fill; cell.font = hfnt
        cell.alignment = hal; cell.border = _tb()
    a1 = PatternFill('solid', fgColor='EBF3FB')
    a2 = PatternFill('solid', fgColor='FFFFFF')
    def _xl_safe(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return ', '.join(f'{x:.4g}' if isinstance(x, (int, float, np.floating)) and pd.notna(x)
                              else str(x) for x in v)
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    for ri, row in df_data.iterrows():
        ws.append([_xl_safe(v) for v in row])
        lr = ws.max_row
        af = a1 if lr % 2 == 0 else a2
        for cell in ws[lr]:
            cell.fill = af
            cell.alignment = Alignment(horizontal='center')
            cell.border = _tb()
            cell.font = Font(size=9, name=TNR)
    ws.auto_filter.ref = ws.dimensions
    for i in range(1, df_data.shape[1]+1):
        ws.column_dimensions[get_column_letter(i)].width = max(
            14, min(24, len(str(df_data.columns[i-1]))+3))

# ── Workbook 1: statistics (Sec. 2.6, 3.1, 3.2) ─────────────────────────────
wb1 = Workbook(); wb1.remove(wb1.active)
write_df(wb1.create_sheet('1_ANOVA_Omnibus'), df_anova, '1F3864')
# ADDED: assumption checks (Shapiro-Wilk, Levene, conditional Kruskal/Welch)
write_df(wb1.create_sheet('2_Assumption_Checks'), df_assum, '833C00')
write_df(wb1.create_sheet('3_Tukey_Force'),
         tukey_all['max_force_N'][['A','B','mean_A','mean_B','diff',
                                    'CI_lo','CI_hi','q','p','sig']],
         'C55A11')
write_df(wb1.create_sheet('4_Tukey_Energy'),
         tukey_all['energy_Nmm'][['A','B','mean_A','mean_B','diff',
                                   'CI_lo','CI_hi','q','p','sig']],
         'C55A11')
write_df(wb1.create_sheet('5_Cohen_d'),
         df_cohen[['group','response','d','CI_lo','CI_hi','mag','pct']],
         '4472C4')
chi_sum = pd.DataFrame({'chi2':[round(chi2_s,3)], 'p':[float(p_chi)],
                         'df':[dof_chi], 'cramers_V':[round(cramv,4)]})
write_df(wb1.create_sheet('6_ChiSquare'), chi_sum, '7030A0')
cld_rows = [{'group': g,
             'n': len(df[df['group']==g]),
             'Force_mean': round(df[df['group']==g]['max_force_N'].mean(),1),
             'Force_SD':   round(df[df['group']==g]['max_force_N'].std(ddof=1),1),
             'Force_CLD':  cld_force[g],
             'Energy_mean':round(df[df['group']==g]['energy_Nmm'].mean(),1),
             'Energy_SD':  round(df[df['group']==g]['energy_Nmm'].std(ddof=1),1),
             'Energy_CLD': cld_energy[g]}
            for g in GROUP_ORDER]
write_df(wb1.create_sheet('7_CLD_Summary'), pd.DataFrame(cld_rows), '375623')
# ADDED: two-way ANOVA (theta x IL) and per-angle one-way ANOVA on IL (Sec. 3.2)
write_df(wb1.create_sheet('8_TwoWay_ANOVA'), df_twoway, '1F3864')
write_df(wb1.create_sheet('9_OneWay_IL_perAngle'), df_oneway_il, '1F3864')
# ADDED: bootstrap AI-activation-threshold CI (Sec. 2.6 / 3.1)
write_df(wb1.create_sheet('10_AI_Threshold_Bootstrap'), df_bootstrap, '833C00')
wb1.save(STATS_XLS); print(f"  -> {STATS_XLS}")

# ── Workbook 2: ML pipeline (Sec. 4.1-4.3) ──────────────────────────────────
wb2 = Workbook(); wb2.remove(wb2.active)
write_df(wb2.create_sheet('1_CV_Results'), df_cv)
# ADDED: the tuned hyperparameters actually used for the final models
tuned_rows = [dict(model=mn, param=p, value=str(v))
              for mn, params in TUNED_PARAMS.items() for p, v in params.items()]
write_df(wb2.create_sheet('2_Tuned_Hyperparams'), pd.DataFrame(tuned_rows), '2E75B6')
# ADDED: train-vs-CV R^2 overfitting-gap check (Sec. 4.3)
write_df(wb2.create_sheet('3_Overfit_Check'), df_overfit, '833C00')
# ADDED: ensemble weights actually used (Sec. 4.1)
ens_rows = pd.DataFrame([{'target': PRIMARY_TARGET, 'model': mn, 'weight': round(w,4)}
                          for mn, w in ENSEMBLE_W.items()])
write_df(wb2.create_sheet('4_Ensemble_Weights'), ens_rows, '2E75B6')
write_df(wb2.create_sheet('5_Classification', ), df_cls, '375623')
write_df(wb2.create_sheet('6_Feature_Imp'), df_fi,   'C55A11')
write_df(wb2.create_sheet('7_SHAP'),        df_shap, '7030A0')
write_df(wb2.create_sheet('8_BO_Results'),  df_bo,   'B22222', 'FFFFFF')
# ADDED: physical-validation comparison table (Sec. 4.5 / 5.1, Table 5)
if len(df_val):
    write_df(wb2.create_sheet('9_Validation_vs_Predicted'), df_validation, 'B22222', 'FFFFFF')
wb2.save(ML_XLS); print(f"  -> {ML_XLS}")

# ── Download all outputs in Colab ───────────────────────────────────────────
if COLAB:
    import shutil
    _all_outputs = [ANALYSIS_PDF, THRESHOLD_PDF, SUPP_PDF, ML_PDF, DESIGN_PDF,
                     STATS_XLS, ML_XLS]
    if len(df_val):
        _all_outputs.append(VALIDATION_PDF)
    for f in _all_outputs:
        files.download(f)
        try:
            shutil.copy(f, f'/content/drive/MyDrive/{f}')
        except Exception:
            pass
    print("\nFiles downloaded and saved to Google Drive")

print("\n" + "="*65)
print("COMPLETE -- SLJ_Analysis_Pro.py  v10.0")
print("="*65)
print(f"  {ANALYSIS_PDF:<28} -- Figs 1-7    (statistical analysis)")
print(f"  {THRESHOLD_PDF:<28} -- Fig 5-style AI-threshold plot + bootstrap")
print(f"  {SUPP_PDF:<28} -- Supplementary (Tukey HSD matrix)")
print(f"  {ML_PDF:<28} -- Figs 8-15   (ML + SHAP + BO)")
print(f"  {DESIGN_PDF:<28} -- Figs 16-18  (design space)")
if len(df_val):
    print(f"  {VALIDATION_PDF:<28} -- Physical validation (Figs 11-12 style)")
print(f"  {STATS_XLS:<28} -- ANOVA, assumption checks, two-way ANOVA, Tukey,")
print(f"  {'':<28}    Cohen's d, chi-square, bootstrap CI")
print(f"  {ML_XLS:<28} -- CV, tuned hyperparameters, SHAP, BO, validation")
print("="*65)
print("\nCOLAB UPLOAD REQUIRED:  your SLJ ML database .xlsx  (only this file)")
print("="*65)
