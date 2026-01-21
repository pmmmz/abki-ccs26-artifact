# import os
# import csv
# import json
# import numpy as np
# import datetime as dt
# from zoneinfo import ZoneInfo
# from typing import List, Tuple

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# """
# KMeans pipeline with TRUE timestamp alignment + fixed-length resampling,
# PLUS exporting an enriched key-log TSV with a stable index and per-event
# cluster assignment.

# What this script adds compared to previous kmean.py:
#   - Adds an incremental index (1..N) for every row in the original key log
#   - Performs timestamp-aligned slicing and clustering only on events that
#     intersect captured frames (>=2 frames after padding window)
#   - Writes an enriched TSV with columns:
#         Index, Arrived, Released, Key, ArrivedEpoch, ReleasedEpoch,
#         Sliced(0/1), ClusterLabel(0..K-1), ClusterName(cluster_#), i0, i1, UserOrder
#     *UserOrder* is left blank for you to fill later to control custom input order
#   - Random naming of clusters to cluster_1..cluster_K (stable via seed)
#   - Saves the label→name mapping as JSON for reuse

# Adjust CONFIG below for your paths and preferences.
# """

# # ======================== CONFIG ========================
# V_PATH = "vmatrix/10_25/V_10_25_1_21.npy"
# TS_PATH = "vmatrix/10_25/timestamps_10_25_1_21.npy"
# KEY_LOG_PATH = "keystroke_time/10_25/key_log_both_25_1.txt"  # Input TSV

# # Output enriched TSV + cluster map
# OUTPUT_ENRICHED_TSV = "key_log_both_25_1_enriched.tsv"
# OUTPUT_CLUSTER_MAP_JSON = "cluster_name_map.json"

# TIMEZONE = "America/Los_Angeles"

# # slicing window around each keystroke
# LEAD_SEC = 0.02   # include a little context before key arrived
# LAG_SEC  = 0.10   # include a little context after key released

# # normalize segment length via linear interpolation
# TARGET_LEN = 32

# # If there is a constant offset between key log clock and capture clock
# CLOCK_OFFSET_SEC = 0.0  # e.g., +0.35 means shift all key times 0.35s later

# # Dimensionality reduction for visualization (and clustering stability)
# USE_PCA_BEFORE_TSNE = True
# PCA_N_COMPONENTS = 50
# TSNE_PERPLEXITY = 30
# TSNE_LEARNING_RATE = "auto"

# # KMeans config
# DESIRED_N_CLUSTERS = 37  # 26 letters + space
# KMEANS_RANDOM_STATE = 0

# # Random naming of clusters (label -> cluster_#), keep stable via seed
# CLUSTER_NAME_RANDOM_SEED = 2025

# # Save plot
# SAVE_FIG_PATH = "kmeans_tsne.png"

# # ========================================================

# def load_V(V_path: str) -> np.ndarray:
#     V = np.load(V_path, allow_pickle=True)
#     if isinstance(V, np.ndarray) and V.dtype == object:
#         V = np.stack(list(V), axis=0)
#     return V


# def load_timestamps(ts_path: str) -> np.ndarray:
#     ts = np.load(ts_path)
#     ts = np.asarray(ts, dtype=float)
#     return ts


# def parse_key_log_with_epoch(path: str, tz_name: str) -> List[Tuple[str, str, str, float, float]]:
#     """Return list of (arrived_str, released_str, key_name, arrived_epoch, released_epoch).
#        Accept header, with or without microseconds.
#     """
#     tz = ZoneInfo(tz_name)
#     out = []
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         reader = csv.reader(f, delimiter='\t')
#         first = next(reader, None)
#         def _is_dt(s: str) -> bool:
#             try:
#                 dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f"); return True
#             except Exception:
#                 try:
#                     dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"); return True
#                 except Exception:
#                     return False
#         def _parse_one(s: str) -> dt.datetime:
#             try:
#                 return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")
#             except ValueError:
#                 return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
#         # header check
#         if first is not None and not _is_dt(first[0]):
#             pass  # skip header
#         else:
#             if first is not None:
#                 arrived_str, released_str, key_name = first[0].strip(), first[1].strip(), first[2].strip()
#                 a = _parse_one(arrived_str).replace(tzinfo=tz).timestamp()
#                 r = _parse_one(released_str).replace(tzinfo=tz).timestamp()
#                 out.append((arrived_str, released_str, key_name, a, r))
#         for row in reader:
#             if not row or len(row) < 3:
#                 continue
#             arrived_str, released_str, key_name = row[0].strip(), row[1].strip(), row[2].strip()
#             try:
#                 a = _parse_one(arrived_str).replace(tzinfo=tz).timestamp()
#                 r = _parse_one(released_str).replace(tzinfo=tz).timestamp()
#                 out.append((arrived_str, released_str, key_name, a, r))
#             except Exception:
#                 continue
#     return out


# def idx_slice(ts_array: np.ndarray, t0: float, t1: float) -> Tuple[int, int]:
#     i0 = int(np.searchsorted(ts_array, t0, side="left"))
#     i1 = int(np.searchsorted(ts_array, t1, side="right"))
#     return i0, i1


# def resample_linear(seg: np.ndarray, target_len: int) -> np.ndarray:
#     L = seg.shape[0]
#     if L == target_len:
#         return seg
#     if L < 2:
#         return np.repeat(seg, target_len, axis=0)[:target_len]
#     x = np.linspace(0, L - 1, num=target_len)
#     x0 = np.floor(x).astype(int)
#     x1 = np.clip(x0 + 1, 0, L - 1)
#     w = (x - x0).reshape(-1, 1, 1, 1)
#     return (1 - w) * seg[x0] + w * seg[x1]


# def main():
#     # 1) Load data
#     V = load_V(V_PATH)
#     ts = load_timestamps(TS_PATH)
#     if V.shape[0] != ts.shape[0]:
#         m = min(V.shape[0], ts.shape[0])
#         print(f"[WARN] Length mismatch V({V.shape[0]}) vs ts({ts.shape[0]}). Trimming to {m}.")
#         V = V[:m]
#         ts = ts[:m]

#     print(f"Loaded V shape: {V.shape}; timestamps: {ts.shape}")

#     # 2) Parse full key log and apply global offset
#     events = parse_key_log_with_epoch(KEY_LOG_PATH, TIMEZONE)
#     # Assign stable indices to ALL events as they appear in the file
#     # Each item: (idx, arrived_str, released_str, key, a_epoch, r_epoch)
#     indexed_events = []
#     for i, (astr, rstr, key, a, r) in enumerate(events, start=1):
#         indexed_events.append([i, astr, rstr, key, a + CLOCK_OFFSET_SEC, r + CLOCK_OFFSET_SEC])

#     print(f"Parsed {len(indexed_events)} events from {KEY_LOG_PATH}")

#     # 3) Slice by timestamps; record which indices are sliced
#     sliced_segments: List[np.ndarray] = []
#     sliced_event_indices: List[int] = []  # indices into indexed_events (1-based ids)
#     i0_list: List[int] = []
#     i1_list: List[int] = []

#     for (idx, astr, rstr, key, a, r) in indexed_events:
#         t0, t1 = a - LEAD_SEC, r + LAG_SEC
#         i0, i1 = idx_slice(ts, t0, t1)
#         if i1 - i0 >= 2:
#             seg = V[i0:i1]
#             sliced_segments.append(seg)
#             sliced_event_indices.append(idx)
#             i0_list.append(i0)
#             i1_list.append(i1)
#         else:
#             i0_list.append(i0)
#             i1_list.append(i1)

#     print(f"Sliced {len(sliced_segments)}/{len(indexed_events)} events (>=2 frames)")
#     if not sliced_segments:
#         raise RuntimeError("No events sliced. Adjust LEAD_SEC/LAG_SEC or CLOCK_OFFSET_SEC.")

#     # 4) Resample to fixed length
#     segs_fixed = [resample_linear(seg, TARGET_LEN) for seg in sliced_segments]
#     print(f"Per-event resampled shape: {segs_fixed[0].shape}")

#     # 5) Build features (real+imag), flatten, scale
#     X = []
#     for seg in segs_fixed:
#         feat = np.stack([seg.real, seg.imag], axis=-1)
#         X.append(feat.reshape(-1))
#     X = np.stack(X, axis=0)
#     X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

#     # 6) Optional PCA before t-SNE (for visualization)
#     Z_in = X
#     if USE_PCA_BEFORE_TSNE:
#         pca = PCA(n_components=min(PCA_N_COMPONENTS, X.shape[1] - 1))
#         Z_in = pca.fit_transform(X)
#         print(f"After PCA: {Z_in.shape}")

#     tsne = TSNE(n_components=2, init="random", perplexity=TSNE_PERPLEXITY,
#                 learning_rate=TSNE_LEARNING_RATE, random_state=0)
#     Z = tsne.fit_transform(Z_in)

#     # 7) KMeans on embedded space
#     n_clusters = min(DESIRED_N_CLUSTERS, Z.shape[0])
#     if n_clusters < 2:
#         raise RuntimeError("Need at least 2 sliced events for KMeans.")
#     km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=KMEANS_RANDOM_STATE)
#     labels = km.fit_predict(Z)  # length == len(sliced_event_indices)

#     # Random but stable naming: label -> cluster_# (1..n_clusters)
#     rng = np.random.default_rng(CLUSTER_NAME_RANDOM_SEED)
#     perm = np.arange(1, n_clusters + 1)
#     rng.shuffle(perm)
#     label_to_name = {int(lbl): f"cluster_{int(perm[lbl])}" for lbl in range(n_clusters)}

#     # 8) Prepare enriched TSV rows for ALL events
#     # Columns: Index, Arrived, Released, Key, ArrivedEpoch, ReleasedEpoch,
#     #          Sliced, ClusterLabel, ClusterName, i0, i1, UserOrder
#     enriched_rows = []
#     # Pre-fill defaults
#     label_by_index = {idx: -1 for (idx, *_rest) in indexed_events}
#     name_by_index = {idx: "" for (idx, *_rest) in indexed_events}

#     # assign labels/names only for sliced events, preserving event order
#     for idx, lbl in zip(sliced_event_indices, labels):
#         label_by_index[idx] = int(lbl)
#         name_by_index[idx] = label_to_name[int(lbl)]

#     # Build rows
#     for (j, (idx, astr, rstr, key, a, r)) in enumerate(indexed_events):
#         sliced_flag = 1 if idx in sliced_event_indices else 0
#         i0 = i0_list[j]
#         i1 = i1_list[j]
#         row = [
#             idx,
#             astr,
#             rstr,
#             key,
#             label_by_index[idx],
#             name_by_index[idx]
#         ]
#         enriched_rows.append(row)

#     # 9) Write enriched TSV
#     header = [
#         "Index", "Arrived", "Released", "Key", "ClusterLabel", "ClusterName"
#     ]
#     with open(OUTPUT_ENRICHED_TSV, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f, delimiter='\t')
#         writer.writerow(header)
#         writer.writerows(enriched_rows)
#     print(f"Saved enriched TSV → {OUTPUT_ENRICHED_TSV}  (rows={len(enriched_rows)})")

#     # 10) Save cluster name mapping for reuse
#     with open(OUTPUT_CLUSTER_MAP_JSON, "w", encoding="utf-8") as f:
#         json.dump({str(k): v for k, v in label_to_name.items()}, f, ensure_ascii=False, indent=2)
#     print(f"Saved cluster-name map → {OUTPUT_CLUSTER_MAP_JSON}")

#     # 11) Plot t-SNE colored by KMeans label
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(Z[:, 0], Z[:, 1], s=14, c=labels)
#     plt.title("t-SNE of V-segments (timestamp-aligned), colored by KMeans label")
#     plt.colorbar(sc, label="ClusterLabel")
#     plt.tight_layout()
#     try:
#         plt.savefig(SAVE_FIG_PATH, dpi=180)
#         print(f"Saved figure to {SAVE_FIG_PATH}")
#     except Exception as e:
#         print(f"[WARN] Could not save figure: {e}")
#     plt.show()

#     # Summary
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     print("\n=== Summary ===")
#     print(f"Events total: {len(indexed_events)} | sliced: {len(sliced_event_indices)} | clusters: {n_clusters}")
#     for lbl, cnt in zip(unique_labels, counts):
#         print(f"  label {lbl:2d} -> {label_to_name[int(lbl)]:>10s}: {cnt} events")


# if __name__ == "__main__":
#     main()














import os
import csv
import json
import numpy as np
import datetime as dt
from zoneinfo import ZoneInfo
from typing import List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score  # ← 新增：用于计算轮廓系数
import matplotlib.pyplot as plt

"""
KMeans pipeline with TRUE timestamp alignment + fixed-length resampling,
PLUS exporting an enriched key-log TSV with a stable index and per-event
cluster assignment.

What this script adds compared to previous kmean.py:
  - Adds an incremental index (1..N) for every row in the original key log
  - Performs timestamp-aligned slicing and clustering only on events that
    intersect captured frames (>=2 frames after padding window)
  - Writes an enriched TSV with columns:
        Index, Arrived, Released, Key, ArrivedEpoch, ReleasedEpoch,
        Sliced(0/1), ClusterLabel(0..K-1), ClusterName(cluster_#), i0, i1, UserOrder
    *UserOrder* is left blank for you to fill later to control custom input order
  - Random naming of clusters to cluster_1..cluster_K (stable via seed)
  - Saves the label→name mapping as JSON for reuse

Adjust CONFIG below for your paths and preferences.
"""

# ======================== CONFIG ========================
V_PATH = "vmatrix/10_25/V_10_25_1_21.npy"
TS_PATH = "vmatrix/10_25/timestamps_10_25_1_21.npy"
KEY_LOG_PATH = "keystroke_time/10_25/key_log_both_25_1.txt"  # Input TSV

# Output enriched TSV + cluster map
OUTPUT_ENRICHED_TSV = "key_log_both_25_1_enriched.tsv"
OUTPUT_CLUSTER_MAP_JSON = "cluster_name_map.json"

TIMEZONE = "America/Los_Angeles"

# slicing window around each keystroke
LEAD_SEC = 0.02   # include a little context before key arrived
LAG_SEC  = 0.10   # include a little context after key released

# normalize segment length via linear interpolation
TARGET_LEN = 32

# If there is a constant offset between key log clock and capture clock
CLOCK_OFFSET_SEC = 0.0  # e.g., +0.35 means shift all key times 0.35s later

# Dimensionality reduction for visualization (and clustering stability)
USE_PCA_BEFORE_TSNE = True
PCA_N_COMPONENTS = 50
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = "auto"

# KMeans config
DESIRED_N_CLUSTERS = 37  # 26 letters + space
KMEANS_RANDOM_STATE = 0

# Random naming of clusters (label -> cluster_#), keep stable via seed
CLUSTER_NAME_RANDOM_SEED = 2025

# Save plot
SAVE_FIG_PATH = "kmeans_tsne.png"

# ========================================================

def load_V(V_path: str) -> np.ndarray:
    V = np.load(V_path, allow_pickle=True)
    if isinstance(V, np.ndarray) and V.dtype == object:
        V = np.stack(list(V), axis=0)
    return V


def load_timestamps(ts_path: str) -> np.ndarray:
    ts = np.load(ts_path)
    ts = np.asarray(ts, dtype=float)
    return ts


def parse_key_log_with_epoch(path: str, tz_name: str) -> List[Tuple[str, str, str, float, float]]:
    """Return list of (arrived_str, released_str, key_name, arrived_epoch, released_epoch).
       Accept header, with or without microseconds.
    """
    tz = ZoneInfo(tz_name)
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter='\t')
        first = next(reader, None)
        def _is_dt(s: str) -> bool:
            try:
                dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f"); return True
            except Exception:
                try:
                    dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"); return True
                except Exception:
                    return False
        def _parse_one(s: str) -> dt.datetime:
            try:
                return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        # header check
        if first is not None and not _is_dt(first[0]):
            pass  # skip header
        else:
            if first is not None:
                arrived_str, released_str, key_name = first[0].strip(), first[1].strip(), first[2].strip()
                a = _parse_one(arrived_str).replace(tzinfo=tz).timestamp()
                r = _parse_one(released_str).replace(tzinfo=tz).timestamp()
                out.append((arrived_str, released_str, key_name, a, r))
        for row in reader:
            if not row or len(row) < 3:
                continue
            arrived_str, released_str, key_name = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                a = _parse_one(arrived_str).replace(tzinfo=tz).timestamp()
                r = _parse_one(released_str).replace(tzinfo=tz).timestamp()
                out.append((arrived_str, released_str, key_name, a, r))
            except Exception:
                continue
    return out


def idx_slice(ts_array: np.ndarray, t0: float, t1: float) -> Tuple[int, int]:
    i0 = int(np.searchsorted(ts_array, t0, side="left"))
    i1 = int(np.searchsorted(ts_array, t1, side="right"))
    return i0, i1


def resample_linear(seg: np.ndarray, target_len: int) -> np.ndarray:
    L = seg.shape[0]
    if L == target_len:
        return seg
    if L < 2:
        return np.repeat(seg, target_len, axis=0)[:target_len]
    x = np.linspace(0, L - 1, num=target_len)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, L - 1)
    w = (x - x0).reshape(-1, 1, 1, 1)
    return (1 - w) * seg[x0] + w * seg[x1]


def main():
    # 1) Load data
    V = load_V(V_PATH)
    ts = load_timestamps(TS_PATH)
    if V.shape[0] != ts.shape[0]:
        m = min(V.shape[0], ts.shape[0])
        print(f"[WARN] Length mismatch V({V.shape[0]}) vs ts({ts.shape[0]}). Trimming to {m}.")
        V = V[:m]
        ts = ts[:m]

    print(f"Loaded V shape: {V.shape}; timestamps: {ts.shape}")

    # 2) Parse full key log and apply global offset
    events = parse_key_log_with_epoch(KEY_LOG_PATH, TIMEZONE)
    # Assign stable indices to ALL events as they appear in the file
    # Each item: (idx, arrived_str, released_str, key, a_epoch, r_epoch)
    indexed_events = []
    for i, (astr, rstr, key, a, r) in enumerate(events, start=1):
        indexed_events.append([i, astr, rstr, key, a + CLOCK_OFFSET_SEC, r + CLOCK_OFFSET_SEC])

    print(f"Parsed {len(indexed_events)} events from {KEY_LOG_PATH}")

    # 3) Slice by timestamps; record which indices are sliced
    sliced_segments: List[np.ndarray] = []
    sliced_event_indices: List[int] = []  # indices into indexed_events (1-based ids)
    i0_list: List[int] = []
    i1_list: List[int] = []

    for (idx, astr, rstr, key, a, r) in indexed_events:
        t0, t1 = a - LEAD_SEC, r + LAG_SEC
        i0, i1 = idx_slice(ts, t0, t1)
        if i1 - i0 >= 2:
            seg = V[i0:i1]
            sliced_segments.append(seg)
            sliced_event_indices.append(idx)
            i0_list.append(i0)
            i1_list.append(i1)
        else:
            i0_list.append(i0)
            i1_list.append(i1)

    print(f"Sliced {len(sliced_segments)}/{len(indexed_events)} events (>=2 frames)")
    if not sliced_segments:
        raise RuntimeError("No events sliced. Adjust LEAD_SEC/LAG_SEC or CLOCK_OFFSET_SEC.")

    # 4) Resample to fixed length
    segs_fixed = [resample_linear(seg, TARGET_LEN) for seg in sliced_segments]
    print(f"Per-event resampled shape: {segs_fixed[0].shape}")

    # 5) Build features (real+imag), flatten, scale
    X = []
    for seg in segs_fixed:
        feat = np.stack([seg.real, seg.imag], axis=-1)
        X.append(feat.reshape(-1))
    X = np.stack(X, axis=0)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # 6) Optional PCA before t-SNE (for visualization)
    Z_in = X
    if USE_PCA_BEFORE_TSNE:
        pca = PCA(n_components=min(PCA_N_COMPONENTS, X.shape[1] - 1))
        Z_in = pca.fit_transform(X)
        print(f"After PCA: {Z_in.shape}")

    tsne = TSNE(
        n_components=2,
        init="random",
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LEARNING_RATE,
        random_state=0
    )
    Z = tsne.fit_transform(Z_in)

    # 7) KMeans on embedded space
    n_clusters = min(DESIRED_N_CLUSTERS, Z.shape[0])
    if n_clusters < 2:
        raise RuntimeError("Need at least 2 sliced events for KMeans.")
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=KMEANS_RANDOM_STATE)
    labels = km.fit_predict(Z)  # length == len(sliced_event_indices)

    # 7.1) Silhouette score on the same embedded space
    # 只有当样本数 > 聚类数时，轮廓系数才比较有意义
    if Z.shape[0] > n_clusters:
        sil_score = silhouette_score(Z, labels)
        print(f"\nGlobal Silhouette score (on t-SNE space): {sil_score:.4f}")
    else:
        sil_score = None
        print("\nNot enough samples to compute a reliable silhouette score (n_samples <= n_clusters).")

    # Random but stable naming: label -> cluster_# (1..n_clusters)
    rng = np.random.default_rng(CLUSTER_NAME_RANDOM_SEED)
    perm = np.arange(1, n_clusters + 1)
    rng.shuffle(perm)
    label_to_name = {int(lbl): f"cluster_{int(perm[lbl])}" for lbl in range(n_clusters)}

    # 8) Prepare enriched TSV rows for ALL events
    # Columns: Index, Arrived, Released, Key, ClusterLabel, ClusterName, i0, i1, UserOrder
    enriched_rows = []
    # Pre-fill defaults
    label_by_index = {idx: -1 for (idx, *_rest) in indexed_events}
    name_by_index = {idx: "" for (idx, *_rest) in indexed_events}

    # assign labels/names only for sliced events, preserving event order
    for idx, lbl in zip(sliced_event_indices, labels):
        label_by_index[idx] = int(lbl)
        name_by_index[idx] = label_to_name[int(lbl)]

    # Build rows
    for (j, (idx, astr, rstr, key, a, r)) in enumerate(indexed_events):
        # sliced_flag = 1 if idx in sliced_event_indices else 0  # 如需可再加回列
        i0 = i0_list[j]
        i1 = i1_list[j]
        row = [
            idx,
            astr,
            rstr,
            key,
            label_by_index[idx],
            name_by_index[idx]
        ]
        enriched_rows.append(row)

    # 9) Write enriched TSV
    header = [
        "Index", "Arrived", "Released", "Key", "ClusterLabel", "ClusterName"
    ]
    with open(OUTPUT_ENRICHED_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(enriched_rows)
    print(f"Saved enriched TSV → {OUTPUT_ENRICHED_TSV}  (rows={len(enriched_rows)})")

    # 10) Save cluster name mapping for reuse
    with open(OUTPUT_CLUSTER_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_to_name.items()}, f, ensure_ascii=False, indent=2)
    print(f"Saved cluster-name map → {OUTPUT_CLUSTER_MAP_JSON}")

    # 11) Plot t-SNE colored by KMeans label
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], s=14, c=labels)
    plt.title("t-SNE of V-segments (timestamp-aligned), colored by KMeans label")
    plt.colorbar(sc, label="ClusterLabel")
    plt.tight_layout()
    try:
        plt.savefig(SAVE_FIG_PATH, dpi=180)
        print(f"Saved figure to {SAVE_FIG_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save figure: {e}")
    plt.show()

    # Summary
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n=== Summary ===")
    print(f"Events total: {len(indexed_events)} | sliced: {len(sliced_event_indices)} | clusters: {n_clusters}")
    if sil_score is not None:
        print(f"Silhouette score: {sil_score:.4f}")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  label {lbl:2d} -> {label_to_name[int(lbl)]:>10s}: {cnt} events")


if __name__ == "__main__":
    main()
