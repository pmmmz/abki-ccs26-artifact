import os
import ast
import math
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Hashable, Optional
import cluster_lookup  


# ---------- Config: distance-table paths ----------
CLUSTER_CSV = "./outputs/cluster_distance_26x26.csv"
LETTER_CSV  = "./outputs/letter_distance_26x26.csv"


# ---------- Load a square distance table ----------
def _load_square_df(path: str) -> pd.DataFrame:
    """
    Load a square distance table from CSV:
    - strip and normalize row/column names as strings
    - coerce values to numeric
    """
    df = pd.read_csv(path, header=0, index_col=0)
    df.index = [str(x).strip() for x in df.index]
    df.columns = [str(x).strip() for x in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# ---------- Label sequence -> distance matrix (cluster distance table) ----------
def labels_to_distance_matrix(labels, cluster_df: pd.DataFrame) -> np.ndarray:
    """
    Convert a cluster-label sequence into an NxN distance matrix
    using the cluster distance table.
    """
    n = len(labels)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0.0
            else:
                ri, cj = str(labels[i]), str(labels[j])
                try:
                    M[i, j] = float(cluster_df.loc[ri, cj])
                except KeyError:
                    M[i, j] = float(cluster_df.loc[int(ri), int(cj)])
    return M


# ---------- Word -> distance matrix (letter distance table) ----------
def word_to_letter_distance_matrix(word: str, letter_df: pd.DataFrame) -> np.ndarray:
    """
    Convert a word into an NxN distance matrix using the letter distance table.
    """
    w = word.lower()
    n = len(w)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            M[i, j] = 0.0 if i == j else float(letter_df.loc[w[i], w[j]])
    return M


# ---------- Matrix similarity (upper-tri MAE / Frobenius) ----------
def matrix_distance_score(A: np.ndarray, B: np.ndarray, mode: str = "mae_upper") -> float:
    """
    Compute matrix distance between A and B:
    - mae_upper: mean absolute error on the upper triangle (k=1)
    - fro: Frobenius norm
    """
    assert A.shape == B.shape, "Matrix shapes must match"
    n = A.shape[0]
    if mode == "mae_upper":
        idx = np.triu_indices(n, k=1)
        return float(np.mean(np.abs(A[idx] - B[idx])))
    elif mode == "fro":
        return float(np.linalg.norm(A - B, ord="fro"))
    else:
        raise ValueError("Unsupported mode")


# ---------- Rank candidate words for one group ----------
def rank_candidates_by_distance(label_group, candidates, 
                                cluster_csv=CLUSTER_CSV,
                                letter_csv=LETTER_CSV,
                                score_mode="mae_upper"):
    """
    For a single group (a label sequence), rank candidate words by
    distance-matrix similarity:
    - build reference matrix from cluster labels
    - build candidate matrices from letters
    - score and sort
    """
    cluster_df = _load_square_df(cluster_csv)
    letter_df = _load_square_df(letter_csv)

    refM = labels_to_distance_matrix(label_group, cluster_df)
    rows = []
    for w in candidates:
        M = word_to_letter_distance_matrix(w, letter_df)
        if M.shape != refM.shape:
            score = float("inf")
        else:
            score = matrix_distance_score(refM, M, mode=score_mode)
        rows.append({"word": w, "score": score})
    rows.sort(key=lambda x: x["score"])
    return rows


# ========= Utilities =========
def generate_relationship_matrix_from_tokens(tokens: List[Hashable]) -> np.ndarray:
    """
    Build an equality-relationship matrix:
    M[i,j]=1 if tokens[i]==tokens[j] else 0.
    """
    n = len(tokens)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            M[i, j] = 1 if tokens[i] == tokens[j] else 0
    return M


def label_token_groups(groups: List[List[Hashable]]) -> List[List[str]]:
    """
    Assign display-only global labels (L1,L2,...) to tokens across groups.
    This is only for printing/visualization, not for decision logic.
    """
    label_map: Dict[Hashable, str] = {}
    next_id = 1
    out: List[List[str]] = []
    for g in groups:
        lab = []
        for t in g:
            if t not in label_map:
                label_map[t] = f"L{next_id}"
                next_id += 1
            lab.append(label_map[t])
        out.append(lab)
    return out


def find_matching_words(input_matrix: np.ndarray, csv_file: str) -> List[str]:
    """
    Look up candidate words from word_groups.csv by matching the flattened
    relationship matrix.
    """
    flat = tuple(input_matrix.flatten().tolist())
    df = pd.read_csv(csv_file)
    assert "Matrix" in df.columns and "Words" in df.columns, "word_groups.csv must contain Matrix / Words columns"
    matches = []
    for _, row in df.iterrows():
        try:
            mat_list = ast.literal_eval(str(row["Matrix"]))
        except Exception:
            continue
        if tuple(mat_list) == flat:
            matches.extend([w.strip() for w in str(row["Words"]).split(",") if w.strip()])
    return matches


def label_word(word: str,
               existing_labels: Dict[str, str] = None,
               next_id: int = 1) -> Tuple[List[str], Dict[str, str], int]:
    """
    Label letters of a word with dynamic IDs (L1,L2,...) based on first occurrence.
    Keeps/extends an existing mapping when provided (for joint decoding).
    """
    if existing_labels is None:
        existing_labels = {}
    labs = []
    for ch in word:
        if ch not in existing_labels:
            existing_labels[ch] = f"L{next_id}"
            next_id += 1
        labs.append(existing_labels[ch])
    return labs, existing_labels, next_id


def expected_labels_for_indices_dynamic(label_groups, indices):
    """
    Given selected group indices, compute the expected flattened label sequence
    under a dynamic re-labeling scheme (L1,L2,... as new tokens appear).
    Returns:
    - flat expected labels for all included groups (concatenated)
    - per-group expected label lists
    """
    label_map = {}
    next_id = 1
    flat = []
    per_group = []

    for idx in indices:
        labs = []
        for t in label_groups[idx]:
            if t not in label_map:
                label_map[t] = f"L{next_id}"
                next_id += 1
            labs.append(label_map[t])
            flat.append(label_map[t])
        per_group.append(labs)

    return flat, per_group


# ===== Joint demodulation (wrapped, with per-step counting) =====
def run_joint_demodulation(per_group_candidates: List[List[str]],
                           label_groups: List[List[int]]) -> Tuple[List[List[str]], List[int], List[int], List[int]]:
    """
    Perform joint decoding across groups:
    - sequentially add groups
    - enforce consistency with expected label sequence
    - allow skipping groups with no viable candidates
    - track number of unique combinations after each step
    """

    def unique_len(combos: List[List[str]]) -> int:
        """Count unique combinations (order-preserving uniqueness check)."""
        seen = set()
        cnt = 0
        for c in combos:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                cnt += 1
        return cnt

    n = len(label_groups)
    included_idxs: List[int] = []
    skipped_idxs: List[int] = []
    matched_combinations: List[List[str]] = []
    step_counts: List[int] = []

    for i in range(n):
        cand = per_group_candidates[i]

        # This group has zero candidates -> skip it
        if not cand:
            skipped_idxs.append(i)
            step_counts.append(unique_len(matched_combinations))
            continue

        if not matched_combinations:
            # Initialize combinations using only this group
            expected_single, _ = expected_labels_for_indices_dynamic(label_groups, [i])
            starters: List[List[str]] = []
            for w in cand:
                labs, _, _ = label_word(w)  # L1,L2,...
                if labs == expected_single:
                    starters.append([w])
            if starters:
                matched_combinations = starters
                included_idxs.append(i)
            else:
                skipped_idxs.append(i)
            step_counts.append(unique_len(matched_combinations))
            continue

        target_idxs = included_idxs + [i]
        expected_flat, _ = expected_labels_for_indices_dynamic(label_groups, target_idxs)

        new_list: List[List[str]] = []
        for combo in matched_combinations:
            # Recompute internal label mapping for the existing combo
            lm, nid = {}, 1
            combo_labs: List[str] = []
            for w in combo:
                l, lm, nid = label_word(w, existing_labels=lm, next_id=nid)
                combo_labs += l

            # Try appending each candidate word for this group
            for w_next in cand:
                l_next, lm2, nid2 = label_word(w_next, existing_labels=dict(lm), next_id=nid)
                if combo_labs + l_next == expected_flat:
                    new_list.append(combo + [w_next])

        if new_list:
            matched_combinations = new_list
            included_idxs.append(i)
        else:
            skipped_idxs.append(i)

        step_counts.append(unique_len(matched_combinations))

    # Final de-duplication (stable order)
    uniq: List[List[str]] = []
    seen = set()
    for combo in matched_combinations:
        t = tuple(combo)
        if t not in seen:
            seen.add(t)
            uniq.append(combo)

    return uniq, included_idxs, skipped_idxs, step_counts


# ========= Save results: a two-row CSV (Baseline / After filtering) =========
def save_two_row_counts_csv(out_path: str, baseline_counts: List[int], filtered_counts: List[int]) -> None:
    """
    Save two sequences as two CSV rows:
    - "Baseline", <counts...>
    - "After filtering", <counts...>
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row1 = ["Baseline"] + [str(x) for x in baseline_counts]
    row2 = ["After filtering"] + [str(x) for x in filtered_counts]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(row1) + "\n")
        f.write(",".join(row2) + "\n")


# ========= Main pipeline (skip-enabled + truncation + baseline comparison + step counts) =========
def process_label_groups_as_words_with_skip(
    words: List[str],
    label_groups: List[List[int]],
    csv_file: str = "word_groups.csv",
    TOPK: Optional[int] = None,
    keep_ratio: float = 0.3,
    keep_min: int = 10,
    save_two_row_csv_path: Optional[str] = None,
):
    """
    For each label group:
    1) build relationship matrix -> retrieve candidates from word_groups.csv
    2) re-rank candidates using distance-matrix similarity
    3) keep full list as baseline, and a filtered list for fast decoding
    4) run joint demodulation on filtered list + baseline list
    5) optionally save per-step combination counts into a two-row CSV
    """

    # Remove invalid labels (-1)
    label_groups = [[t for t in g if t != -1] for g in label_groups]

    # Display-only: global labels for debugging
    display_labels = label_token_groups(label_groups)
    print("Given (global) labels for groups (display only):")
    for i, lab in enumerate(display_labels, 1):
        print(f"Group {i}: [ {' '.join(lab)} ]")

    # ========== Per-group matrices & candidate lists ==========
    per_group_candidates: List[List[str]] = []        
    per_group_candidates_full: List[List[str]] = []   

    def _is_clean_word(w: str, L: int) -> bool:
        """Keep only lowercase alphabetic words with exact length L."""
        w2 = w.lower()
        return len(w2) == L and all('a' <= ch <= 'z' for ch in w2)

    for i, g in enumerate(label_groups, 1):
        M = generate_relationship_matrix_from_tokens(g)
        print(f"\nMatrix for group {i}:")
        print(M)

        # 1) Retrieve candidates by structural matrix matching
        cand = find_matching_words(M, csv_file)

        # De-duplicate + keep only clean words (letters only, length matches)
        cand = list(dict.fromkeys(cand))                    
        cand = [w for w in cand if _is_clean_word(w, len(g))]

        if not cand:
            print("No matching words found for this matrix.")
            per_group_candidates.append([])       
            per_group_candidates_full.append([])  
            continue

        print(f"Words matching this matrix: {', '.join(cand)}")

        # 2) Distance-matrix re-ranking + build (baseline / filtered) candidate lists
        try:
            ranked = rank_candidates_by_distance(
                label_group=g,
                candidates=cand,
                cluster_csv=CLUSTER_CSV,
                letter_csv=LETTER_CSV,
                score_mode="mae_upper"
            )
            print("Top candidates by distance matrix:",
                  ", ".join(f"{r['word']}\u2006({r['score']:.3f})" for r in ranked[:10]))

            # Baseline: keep all ranked candidates
            ranked_full_words = [r["word"] for r in ranked]
            per_group_candidates_full.append(ranked_full_words)

            # Filtered/truncated: ratio + minimum; if TOPK is provided, use TOPK
            if TOPK is None:
                k = max(keep_min, math.ceil(len(ranked) * keep_ratio))
            else:
                k = TOPK
            ranked_for_use = ranked[:k]
            filtered = [r["word"] for r in ranked_for_use]
            per_group_candidates.append(filtered)

        except Exception as e:
            print("Distance-matrix ranking failed:", e)
            # If ranking fails, fall back to raw candidates for both
            per_group_candidates.append(cand)
            per_group_candidates_full.append(cand)

    # ========== Joint demodulation: filtered ==========
    matched_combinations, included_idxs, skipped_idxs, step_counts_filtered = run_joint_demodulation(
        per_group_candidates, label_groups
    )

    # ========== Joint demodulation: baseline (no filtering) ==========
    matched_combinations_full, _, _, step_counts_full = run_joint_demodulation(
        per_group_candidates_full, label_groups
    )

    # ========== Save two-row CSV if requested ==========
    if save_two_row_csv_path is not None:
        L = min(len(step_counts_full), len(step_counts_filtered))
        save_two_row_counts_csv(
            save_two_row_csv_path,
            baseline_counts=step_counts_full[:L],
            filtered_counts=step_counts_filtered[:L],
        )

    # Print results (kept for debugging)
    if not matched_combinations:
        print("\nNo matched word combinations found (even after skipping).")
    else:
        print("\nMatched word combinations (on included groups only):")
        for combo in matched_combinations:
            print(" ".join(combo))
        print(f"\nTotal matched word combinations: {len(matched_combinations)}")

    print(f"\n[Comparison] Baseline (no filtering): {len(matched_combinations_full)} combinations")
    print(f"[Comparison] After filtering:          {len(matched_combinations)} combinations")
    print(f"[Comparison] Reduced from {len(matched_combinations_full)} → {len(matched_combinations)}")

    print("\n[Per-word combination counts] Baseline:")
    for i, v in enumerate(step_counts_full, 1):
        print(f"  After word {i}: {v}")
    print("\n[Per-word combination counts] After filtering:")
    for i, v in enumerate(step_counts_filtered, 1):
        print(f"  After word {i}: {v}")

    if skipped_idxs:
        print(f"\nSkipped groups: {skipped_idxs}")
    else:
        print("\nSkipped groups: None")


# ========= Read sentences from CSV (one sentence per row, one word per cell) =========
def read_sentences_csv(path: str) -> List[List[str]]:
    """Read a CSV where each row is a sentence and each cell is a word."""
    sentences: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            words = [c.strip() for c in row if c and c.strip()]
            if words:
                sentences.append(words)
    return sentences


def sanitize_tag(words: List[str], maxlen: int = 6) -> str:
    """
    Build a short filename tag from words, truncated to avoid overly long filenames.
    """
    tag = "_".join([w.lower() for w in words])
    return tag[:maxlen*len(words)]


def process_sentences_from_csv(
    input_csv: str,
    tsv_path: str,
    csv_word_groups: str = "word_groups.csv",
    TOPK: Optional[int] = None,
    keep_ratio: float = 0.3,
    keep_min: int = 10,
    out_dir: str = "./outputs4",
):
    """
    Batch process sentences from input CSV:
    - convert sentence text -> index groups -> label groups
    - run the skip-enabled + filtering-enabled decoding pipeline
    - save per-step counts to CSV for each sentence
    """
    os.makedirs(out_dir, exist_ok=True)
    sentences = read_sentences_csv(input_csv)
    print(f"[Info] Loaded {len(sentences)} sentence(s) from {input_csv}")

    for idx, words in enumerate(sentences, start=1):
        text = " ".join(words)
        print(f"\n=== Sentence {idx}: {text} ===")

        words_norm, index_groups = cluster_lookup.sentence_to_index_groups(tsv_path, text)
        idx2lab = cluster_lookup.load_index_to_label(tsv_path)
        label_groups = cluster_lookup.groups_indices_to_labels(index_groups, idx2lab)

        tag = sanitize_tag(words, maxlen=8)
        out_counts_csv = os.path.join(out_dir, f"per_word_counts_{idx:02d}_{tag}.csv")

        process_label_groups_as_words_with_skip(
            words_norm,
            label_groups,
            csv_file=csv_word_groups,
            TOPK=TOPK,
            keep_ratio=keep_ratio,
            keep_min=keep_min,
            save_two_row_csv_path=out_counts_csv,
        )
        print(f"[Saved] {out_counts_csv}")


# ---------- One-click entrypoint ----------
if __name__ == "__main__":

    input_sentences_csv = "./inputs/sentences5.csv"
    tsv_path = "./outputs/key_log_both_25_1_enriched_clusterlabel.tsv"

    csv_word_groups = "word_groups.csv"

    TOPK = None
    KEEP_RATIO = 0.3
    KEEP_MIN = 10

    os.makedirs("./inputs", exist_ok=True)
    if os.path.exists("./sentences.csv") and not os.path.exists(input_sentences_csv):
        os.rename("./sentences.csv", input_sentences_csv)

    process_sentences_from_csv(
        input_csv=input_sentences_csv,
        tsv_path=tsv_path,
        csv_word_groups=csv_word_groups,
        TOPK=TOPK,
        keep_ratio=KEEP_RATIO,
        keep_min=KEEP_MIN,
        out_dir="./outputs",
    )
