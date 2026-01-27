import os
import ast
import math
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Hashable, Optional


CLUSTER_CSV = "./inputs/cluster_distance_26x26.csv"
LETTER_CSV  = "./inputs/letter_distance_26x26.csv"


def _load_square_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, index_col=0)
    df.index = [str(x).strip() for x in df.index]
    df.columns = [str(x).strip() for x in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def labels_to_distance_matrix(labels, cluster_df: pd.DataFrame) -> np.ndarray:
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


def word_to_letter_distance_matrix(word: str, letter_df: pd.DataFrame) -> np.ndarray:
    w = word.lower()
    n = len(w)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            M[i, j] = 0.0 if i == j else float(letter_df.loc[w[i], w[j]])
    return M


def matrix_distance_score(A: np.ndarray, B: np.ndarray, mode: str = "mae_upper") -> float:
    assert A.shape == B.shape, "Matrix shapes must match"
    n = A.shape[0]
    if mode == "mae_upper":
        idx = np.triu_indices(n, k=1)
        return float(np.mean(np.abs(A[idx] - B[idx])))
    elif mode == "fro":
        return float(np.linalg.norm(A - B, ord="fro"))
    else:
        raise ValueError("Unsupported mode")


def rank_candidates_by_distance(label_group, candidates,
                                cluster_csv=CLUSTER_CSV,
                                letter_csv=LETTER_CSV,
                                score_mode="mae_upper"):
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


def generate_relationship_matrix_from_tokens(tokens: List[Hashable]) -> np.ndarray:
    n = len(tokens)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            M[i, j] = 1 if tokens[i] == tokens[j] else 0
    return M


def find_matching_words(input_matrix: np.ndarray, csv_file: str) -> List[str]:
    flat = tuple(input_matrix.flatten().tolist())
    df = pd.read_csv(csv_file)
    assert "Matrix" in df.columns and "Words" in df.columns, "word_groups.csv need Matrix / Words "
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

    label_map = {}
    next_id = 1
    flat = []

    for idx in indices:
        for t in label_groups[idx]:
            if t not in label_map:
                label_map[t] = f"L{next_id}"
                next_id += 1
            flat.append(label_map[t])

    return flat


def run_joint_demodulation(per_group_candidates: List[List[str]],
                           label_groups: List[List[int]]) -> Tuple[List[List[str]], List[int], List[int], List[int]]:

    def unique_len(combos: List[List[str]]) -> int:
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

        if not cand:
            skipped_idxs.append(i)
            step_counts.append(unique_len(matched_combinations))
            continue

        if not matched_combinations:
            expected_single = expected_labels_for_indices_dynamic(label_groups, [i])
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
        expected_flat = expected_labels_for_indices_dynamic(label_groups, target_idxs)

        new_list: List[List[str]] = []
        for combo in matched_combinations:
            lm, nid = {}, 1
            combo_labs: List[str] = []
            for w in combo:
                l, lm, nid = label_word(w, existing_labels=lm, next_id=nid)
                combo_labs += l

            for w_next in cand:
                l_next, _, _ = label_word(w_next, existing_labels=dict(lm), next_id=nid)
                if combo_labs + l_next == expected_flat:
                    new_list.append(combo + [w_next])

        if new_list:
            matched_combinations = new_list
            included_idxs.append(i)
        else:
            skipped_idxs.append(i)

        step_counts.append(unique_len(matched_combinations))

    uniq: List[List[str]] = []
    seen = set()
    for combo in matched_combinations:
        t = tuple(combo)
        if t not in seen:
            seen.add(t)
            uniq.append(combo)

    return uniq, included_idxs, skipped_idxs, step_counts


def save_two_row_counts_csv(out_path: str, baseline_counts: List[int], filtered_counts: List[int]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row1 = ["Baseline"] + [str(x) for x in baseline_counts]
    row2 = ["After filtering"] + [str(x) for x in filtered_counts]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(row1) + "\n")
        f.write(",".join(row2) + "\n")


def extract_letter_units_from_newlabel_csv(
    labeled_csv: str,
    cluster_col: str = "Cluster label",
    newlabel_col: str = "NewLabel",
) -> Tuple[pd.DataFrame, List[List[int]], List[List[int]]]:

    df = pd.read_csv(labeled_csv)
    if cluster_col not in df.columns or newlabel_col not in df.columns:
        raise ValueError(f"CSV must contain columns: '{cluster_col}' and '{newlabel_col}'. "
                         f"Got: {list(df.columns)}")

    clusters = df[cluster_col].astype(int).tolist()
    nl = df[newlabel_col].astype(str).tolist()

    label_groups: List[List[int]] = []
    row_groups: List[List[int]] = []

    cur_labels: List[int] = []
    cur_rows: List[int] = []
    cur_is_letter_unit: Optional[bool] = None  # True if current unit is LETTER, False if UNSURE, None if not started

    def flush():
        nonlocal cur_labels, cur_rows, cur_is_letter_unit
        if cur_rows and cur_is_letter_unit is True:
            g = [t for t in cur_labels if t != -1]
            rg = [r for (r, t) in zip(cur_rows, cur_labels) if t != -1]
            if g:
                label_groups.append(g)
                row_groups.append(rg)
        cur_labels, cur_rows, cur_is_letter_unit = [], [], None

    for i in range(len(df)):
        if nl[i].upper() == "SPACE":
            flush()
            continue

        if cur_is_letter_unit is None:
            cur_is_letter_unit = (nl[i].upper() == "LETTER")

        cur_labels.append(clusters[i])
        cur_rows.append(i)

        if cur_is_letter_unit and nl[i].upper() != "LETTER":
            cur_is_letter_unit = False

    flush()
    return df, label_groups, row_groups


def decode_from_newlabel_csv(
    labeled_csv: str,
    word_groups_csv: str = "word_groups.csv",
    out_dir: str = "./outputs_decode",
    out_labeled_csv: str = "./outputs_decode/cluster_labels_decoded.csv",
    TOPK: Optional[int] = None,
    keep_ratio: float = 0.6,
    keep_min: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)

    df, label_groups, row_groups = extract_letter_units_from_newlabel_csv(labeled_csv)
    print(f"[Info] Extracted {len(label_groups)} LETTER keystroke unit(s) from {labeled_csv}")

    words_placeholder = [f"U{i+1}" for i in range(len(label_groups))]

    per_group_candidates: List[List[str]] = []       
    per_group_candidates_full: List[List[str]] = []   

    def _is_clean_word(w: str, L: int) -> bool:
        w2 = w.lower()
        return len(w2) == L and all('a' <= ch <= 'z' for ch in w2)

    for i, g in enumerate(label_groups, start=1):
        M = generate_relationship_matrix_from_tokens(g)

        cand = find_matching_words(M, word_groups_csv)
        cand = list(dict.fromkeys(cand))
        cand = [w for w in cand if _is_clean_word(w, len(g))]

        if not cand:
            print(f"[Unit {i}] No matching words for structure matrix. (len={len(g)})")
            per_group_candidates.append([])
            per_group_candidates_full.append([])
            continue

        try:
            ranked = rank_candidates_by_distance(
                label_group=g,
                candidates=cand,
                cluster_csv=CLUSTER_CSV,
                letter_csv=LETTER_CSV,
                score_mode="mae_upper"
            )
            ranked_full = [r["word"] for r in ranked]
            per_group_candidates_full.append(ranked_full)

            if TOPK is None:
                k = max(keep_min, math.ceil(len(ranked) * keep_ratio))
            else:
                k = TOPK
            per_group_candidates.append([r["word"] for r in ranked[:k]])

            print(f"[Unit {i}] Top by distance:",
                  ", ".join(f"{r['word']}({r['score']:.3f})" for r in ranked[:10]))

        except Exception as e:
            print(f"[Unit {i}] Distance-matrix ranking failed -> fallback. err={e}")
            per_group_candidates.append(cand)
            per_group_candidates_full.append(cand)

    matched, included_idxs, skipped_idxs, step_counts_filtered = run_joint_demodulation(
        per_group_candidates, label_groups
    )
    matched_full, _, _, step_counts_full = run_joint_demodulation(
        per_group_candidates_full, label_groups
    )

    if not matched:
        print("\nNo matched word combinations found (even after skipping).")
    else:
        print("\nMatched word combinations (on included groups only):")
        for combo in matched:
            print(" ".join(combo))
        print(f"\nTotal matched word combinations: {len(matched)}")

    print(f"\n[Comparison] Baseline (no filtering): {len(matched_full)} combinations")
    print(f"[Comparison] After filtering:          {len(matched)} combinations")
    print(f"[Comparison] Reduced from {len(matched_full)} → {len(matched)}")

    print("\n[Per-word combination counts] Baseline:")
    for i in range(min(len(words_placeholder), len(step_counts_full))):
        print(f"  After word {i+1} ({words_placeholder[i]}): {step_counts_full[i]}")
    print("\n[Per-word combination counts] After filtering:")
    for i in range(min(len(words_placeholder), len(step_counts_filtered))):
        print(f"  After word {i+1} ({words_placeholder[i]}): {step_counts_filtered[i]}")

    out_counts_csv = os.path.join(out_dir, "per_word_counts_rows.csv")
    L = min(len(step_counts_full), len(step_counts_filtered))
    save_two_row_counts_csv(out_counts_csv, step_counts_full[:L], step_counts_filtered[:L])
    print(f"\n[Saved] Per-word counts (2 rows): {out_counts_csv}")

    df["DecodedLetter"] = ""  

    if len(matched) == 1:
        combo = matched[0] 
        token2char: Dict[int, str] = {}
        for k, gi in enumerate(included_idxs):
            word = combo[k]
            tokens = label_groups[gi]
            if len(tokens) != len(word):
                continue
            for t, ch in zip(tokens, word):
                if t in token2char and token2char[t] != ch:
                    print("[Warn] token->char conflict detected. This should not happen in a valid unique solution.")
                token2char[t] = ch

        for k, gi in enumerate(included_idxs):
            rows = row_groups[gi]  
            tokens = label_groups[gi]
            for r, t in zip(rows, tokens):
                if t in token2char:
                    df.at[r, "DecodedLetter"] = token2char[t]

        os.makedirs(os.path.dirname(out_labeled_csv), exist_ok=True)
        df.to_csv(out_labeled_csv, index=False, encoding="utf-8-sig")
        print(f"\n[OK] The unique solution has been found and written back to DecodedLetter -> {out_labeled_csv}")

    else:
        print("\nNo unique solution was found.（Matched combinations != 1），not write back to DecodedLetter。")


if __name__ == "__main__":
    labeled_csv = "./cluster_labels_newlabel.csv"   
    word_groups_csv = "word_groups.csv"

    decode_from_newlabel_csv(
        labeled_csv=labeled_csv,
        word_groups_csv=word_groups_csv,
        out_dir="./outputs_decode",
        out_labeled_csv="./outputs_decode/cluster_labels_decoded.csv",
        TOPK=None,
        keep_ratio=1,
        keep_min=10,
    )
