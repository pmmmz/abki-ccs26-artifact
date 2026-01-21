# -*- coding: utf-8 -*-
"""
Pipeline:
PCA (48) → Letter prototypes via GPA → Canonical feature expansion
(distance / cosine / angle + d1 + gap + entropy + top-1 one-hot)

Classifier:
LogisticRegression (L2, multinomial, class_weight='balanced', C=1.5)

Optional:
Lightweight attention add-on (2 scalar features).
If enabled, inner validation selects K / tau.

Goal:
Improve non-letter key accuracy from ~60% to ~70%+
using mild additive canonical features.
"""

import os, csv, json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ======================== Dataset configuration ========================
DATASETS = [
    {"name": "pos_1",  "V_path": "vmatrix/V_8_21_1_1.npy",  "ts_path": "vmatrix/timestamps_8_21_1_1.npy",  "keylog": "keystroke_time/key_log_both_1.txt"},
    {"name": "pos_2",  "V_path": "vmatrix/V_8_21_2_1.npy",  "ts_path": "vmatrix/timestamps_8_21_2_1.npy",  "keylog": "keystroke_time/key_log_both_2.txt"},
    {"name": "pos_3",  "V_path": "vmatrix/V_8_21_3_1.npy",  "ts_path": "vmatrix/timestamps_8_21_3_1.npy",  "keylog": "keystroke_time/key_log_both_3.txt"},
    {"name": "pos_4",  "V_path": "vmatrix/V_8_21_4_1.npy",  "ts_path": "vmatrix/timestamps_8_21_4_1.npy",  "keylog": "keystroke_time/key_log_both_4.txt"},
    {"name": "pos_5",  "V_path": "vmatrix/V_8_21_5_1.npy",  "ts_path": "vmatrix/timestamps_8_21_5_1.npy",  "keylog": "keystroke_time/key_log_both_5.txt"},
    {"name": "pos_6",  "V_path": "vmatrix/V_8_21_6_1.npy",  "ts_path": "vmatrix/timestamps_8_21_6_1.npy",  "keylog": "keystroke_time/key_log_both_6.txt"},
    {"name": "pos_7",  "V_path": "vmatrix/V_8_21_7_1.npy",  "ts_path": "vmatrix/timestamps_8_21_7_1.npy",  "keylog": "keystroke_time/key_log_both_7.txt"},
    {"name": "pos_8",  "V_path": "vmatrix/V_8_21_8_1.npy",  "ts_path": "vmatrix/timestamps_8_21_8_1.npy",  "keylog": "keystroke_time/key_log_both_8.txt"},
    {"name": "pos_9",  "V_path": "vmatrix/V_8_21_9_1.npy",  "ts_path": "vmatrix/timestamps_8_21_9_1.npy",  "keylog": "keystroke_time/key_log_both_9.txt"},
    {"name": "pos_10", "V_path": "vmatrix/V_8_21_10_1.npy", "ts_path": "vmatrix/timestamps_8_21_10_1.npy", "keylog": "keystroke_time/key_log_both_10.txt"},
    {"name": "pos_11", "V_path": "vmatrix/V_8_21_11_1.npy", "ts_path": "vmatrix/timestamps_8_21_11_1.npy", "keylog": "keystroke_time/key_log_both_11.txt"},
]

LEAD_SEC   = 0.02
LAG_SEC    = 0.10
TARGET_LEN = 32


# ======================== Session-level encoding ========================
PCA_DIM          = 48       # target PCA dimension
PCA_MIN_DIM_SAFE = 16       # minimum safe dimension
PCA_SOLVER       = "randomized"


# ======================== GPA alignment ========================
GPA_FIT_MODE = "all_letters"    # or "train_only" for stricter evaluation
GPA_MAX_ITER = 20
GPA_TOL      = 1e-7


# ======================== Canonical entropy temperature ========================
TAU_ENTROPY_ALL = 1.2


# ======================== Optional attention add-on ========================
ATT_ADD_ON = False
GRID_K     = [3, 4, 6]
GRID_TAU   = [0.8, 1.0, 1.3, 1.6]


# ======================== Classifier ========================
LR_RANDOM_STATE = 0
LR_C            = 1.5
DIRICHLET_EPS   = 1e-12


OUTDIR = "ml_outputs_pca48_gpa_canon_plus"
os.makedirs(OUTDIR, exist_ok=True)


# ======================== Basic I/O utilities ========================
def load_V(p):
    """Load V-matrix array."""
    V = np.load(p, allow_pickle=True)
    if isinstance(V, np.ndarray) and V.dtype == object:
        V = np.stack(list(V), axis=0)
    return V


def load_ts(p):
    """Load timestamp array."""
    ts = np.load(p)
    return np.asarray(ts, dtype=float)


def parse_time(s):
    """Parse timestamp string into UNIX time."""
    s = s.strip()
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(s, fmt).timestamp()


def looks_like_time(s):
    """Check whether a string looks like a timestamp."""
    try:
        datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f"); return True
    except:
        try:
            datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"); return True
        except:
            return False


def read_keylog(path):
    """Read keystroke log file with arrival/release timestamps."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f, delimiter="\t")
        first = next(rd, None)
        if first is not None and not looks_like_time(first[0]):
            pass
        else:
            if first is not None:
                try:
                    rows.append((parse_time(first[0]), parse_time(first[1]), first[2].strip()))
                except:
                    pass
        for r in rd:
            try:
                rows.append((parse_time(r[0]), parse_time(r[1]), r[2].strip()))
            except:
                pass
    return rows


def idx_slice(ts, t0, t1):
    """Convert a time window into index range."""
    i0 = int(np.searchsorted(ts, t0, side="left"))
    i1 = int(np.searchsorted(ts, t1, side="right"))
    return i0, i1


def resample_to_len(seg, L):
    """Linearly resample a segment to fixed length L."""
    n = seg.shape[0]
    if n == L: return seg
    if n < 2:  return np.repeat(seg, L, axis=0)[:L]
    x = np.linspace(0, n-1, L)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, n-1)
    w = (x - x0).reshape(-1, *([1]*(seg.ndim-1)))
    return (1 - w) * seg[x0] + w * seg[x1]


def seg_to_vec(seg):
    """Flatten a complex segment into a real-valued vector."""
    real = np.real(seg)
    imag = np.imag(seg)
    feat = np.stack([real, imag], axis=-1)
    return feat.reshape(-1)


def is_letter(k):
    """Check whether a key label is a letter."""
    return (len(k) == 1) and ('a' <= k.lower() <= 'z')


# ======================== Kabsch / GPA ========================
def kabsch(X, Y):
    """Estimate similarity transform (scale, rotation, translation)."""
    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    X0 = X - muX; Y0 = Y - muY
    nX = np.linalg.norm(X0); nY = np.linalg.norm(Y0)
    if nX < 1e-12 or nY < 1e-12:
        return 1.0, np.eye(X.shape[1]), (muY - muX)
    X0 /= nX; Y0 /= nY
    H = X0.T @ Y0
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    s = S.sum()
    t = muY - s * muX @ R
    return s, R, t


def gpa_fit(mats, max_iter=GPA_MAX_ITER, tol=GPA_TOL):
    """Generalized Procrustes Analysis (GPA)."""
    def normalize(Z):
        Zc = Z - Z.mean(axis=0, keepdims=True)
        n = np.linalg.norm(Zc)
        return Zc / max(n, 1e-12)

    canon = normalize(np.mean(np.stack(mats, axis=0), axis=0))
    last = None
    transforms = [None] * len(mats)

    for _ in range(max_iter):
        aligned = []
        for i, X in enumerate(mats):
            s, R, t = kabsch(X, canon)
            transforms[i] = (s, R, t)
            aligned.append(s * X @ R + t)
        aligned = np.stack(aligned, axis=0)
        new_canon = normalize(np.mean(aligned, axis=0))
        obj = float(np.sum((aligned - new_canon) ** 2))
        if last is not None and abs(last - obj) < tol:
            canon = new_canon
            break
        last = obj
        canon = new_canon

    return canon, transforms
