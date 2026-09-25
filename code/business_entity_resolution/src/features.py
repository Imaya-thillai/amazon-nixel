"""Feature Engineering Module for Entity Resolution.

Extracts discriminative string similarity, phonetic, token overlap,
address matching, and interaction features between candidate pairs.
"""

import difflib
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Computes Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / union) if union > 0 else 0.0


def dice_similarity(set_a: set, set_b: set) -> float:
    """Computes Dice coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    denom = len(set_a) + len(set_b)
    return float(2 * intersection / denom) if denom > 0 else 0.0


def get_char_ngrams(text: str, n: int = 3) -> set:
    """Generates set of character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def sequence_matcher_ratio(s1: str, s2: str) -> float:
    """Fast difflib SequenceMatcher similarity ratio (normalized Levenshtein-like)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(difflib.SequenceMatcher(None, s1, s2).quick_ratio())


def token_sort_ratio(s1: str, s2: str) -> float:
    """Token sort ratio: compares alphabetically sorted tokens (word order invariant)."""
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return sequence_matcher_ratio(t1, t2)


def extract_pair_features(
    s1_row: pd.Series,
    cand_row: pd.Series,
    rank: int = 1,
) -> Dict[str, float]:
    """Computes full feature vector for a candidate pair."""
    # Names
    name1 = s1_row["clean_name"]
    name2 = cand_row["clean_name"]
    root1 = s1_row["root_name"]
    root2 = cand_row["root_name"]

    name_tokens1 = set(name1.split())
    name_tokens2 = set(name2.split())

    # Exact matches
    name_exact = 1.0 if name1 == name2 and name1 else 0.0
    root_name_exact = 1.0 if root1 == root2 and root1 else 0.0

    # Token overlap
    name_jaccard_word = jaccard_similarity(name_tokens1, name_tokens2)
    name_dice_word = dice_similarity(name_tokens1, name_tokens2)

    # Character n-grams
    char3_1 = get_char_ngrams(name1, 3)
    char3_2 = get_char_ngrams(name2, 3)
    name_jaccard_char3 = jaccard_similarity(char3_1, char3_2)

    char4_1 = get_char_ngrams(name1, 4)
    char4_2 = get_char_ngrams(name2, 4)
    name_jaccard_char4 = jaccard_similarity(char4_1, char4_2)

    # Sequence alignment & token sort
    name_seq_ratio = sequence_matcher_ratio(name1, name2)
    name_sort_ratio = token_sort_ratio(name1, name2)

    # Length & token counts
    l1, l2 = len(name1), len(name2)
    name_len_diff = abs(l1 - l2) / max(l1, l2, 1)
    word_count_diff = abs(len(name_tokens1) - len(name_tokens2))

    # First word match
    first1 = name1.split()[0] if name1 else ""
    first2 = name2.split()[0] if name2 else ""
    first_word_match = 1.0 if first1 == first2 and first1 else 0.0

    # Addresses
    addr1 = s1_row["clean_address"]
    addr2 = cand_row["clean_address"]
    addr_tokens1 = set(addr1.split())
    addr_tokens2 = set(addr2.split())

    addr_exact = 1.0 if addr1 == addr2 and addr1 else 0.0
    addr_jaccard_word = jaccard_similarity(addr_tokens1, addr_tokens2)
    addr_dice_word = dice_similarity(addr_tokens1, addr_tokens2)
    addr_seq_ratio = sequence_matcher_ratio(addr1, addr2)
    addr_sort_ratio = token_sort_ratio(addr1, addr2)

    # Postal code match
    p1 = s1_row.get("postal_code")
    p2 = cand_row.get("postal_code")
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else 0.0
    elif not p1 and not p2:
        postal_match = 0.5  # Neutral when both missing
    else:
        postal_match = 0.2  # Asymmetric missing

    # Numeric digits overlap
    import re
    d1 = set(re.findall(r"\b\d+\b", addr1))
    d2 = set(re.findall(r"\b\d+\b", addr2))
    digits_jaccard = jaccard_similarity(d1, d2)

    # Interactions
    interaction_score = name_jaccard_word * addr_jaccard_word
    combined_seq_ratio = (name_seq_ratio + addr_seq_ratio) / 2.0

    # Source flags
    cand_id = cand_row["entity_id"]
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0

    return {
        "name_exact": name_exact,
        "root_name_exact": root_name_exact,
        "name_jaccard_word": name_jaccard_word,
        "name_dice_word": name_dice_word,
        "name_jaccard_char3": name_jaccard_char3,
        "name_jaccard_char4": name_jaccard_char4,
        "name_seq_ratio": name_seq_ratio,
        "name_sort_ratio": name_sort_ratio,
        "name_len_diff": name_len_diff,
        "word_count_diff": float(word_count_diff),
        "first_word_match": first_word_match,
        "addr_exact": addr_exact,
        "addr_jaccard_word": addr_jaccard_word,
        "addr_dice_word": addr_dice_word,
        "addr_seq_ratio": addr_seq_ratio,
        "addr_sort_ratio": addr_sort_ratio,
        "postal_match": postal_match,
        "digits_jaccard": digits_jaccard,
        "interaction_score": interaction_score,
        "combined_seq_ratio": combined_seq_ratio,
        "is_source2": is_source2,
        "candidate_rank": float(rank),
    }


def build_feature_dataframe(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Builds a complete feature matrix for all candidate pairs.

    Returns:
        X: DataFrame of feature vectors
        y: Series of labels (1 if match, 0 if negative; None if test set)
        s1_ids: List of Source 1 entity IDs
        cand_ids: List of Candidate entity IDs
    """
    s1_dict = df_s1.set_index("entity_id").to_dict("index")
    s23_dict = df_s23.set_index("entity_id").to_dict("index")

    rows = []
    labels = []
    out_s1 = []
    out_cands = []

    for s1_id, cand_list in candidates.items():
        if s1_id not in s1_dict:
            continue
        s1_row = s1_dict[s1_id]
        true_set = set(ground_truth.get(s1_id, [])) if ground_truth else None

        for rank, cand_id in enumerate(cand_list, start=1):
            if cand_id not in s23_dict:
                continue
            cand_row = s23_dict[cand_id]

            # Reconstruct series-like dict
            s1_data = pd.Series(s1_row)
            cand_data = pd.Series(cand_row)
            cand_data["entity_id"] = cand_id

            feats = extract_pair_features(s1_data, cand_data, rank=rank)
            rows.append(feats)
            out_s1.append(s1_id)
            out_cands.append(cand_id)

            if true_set is not None:
                labels.append(1 if cand_id in true_set else 0)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label") if ground_truth is not None else None
    return X, y, out_s1, out_cands
