"""Official Competition Metric Evaluation for Amazon ML Challenge 2026.

Business Entity Resolution Challenge.
Evaluates Macro-averaged F_beta (beta=0.5) score across all Source 1 entities,
including exact singleton credit and penalty logic.
"""

from typing import Dict, Set, List, Tuple
import numpy as np


def compute_entity_f05(predicted_ids: Set[str], ground_truth_ids: Set[str]) -> float:
    """Computes F_0.5 score for a single Source 1 entity.

    Rules:
    - If ground truth has no matches (singleton):
        - Score = 1.0 if predicted is empty
        - Score = 0.0 if any match predicted
    - If ground truth has matches:
        - Score = 0.0 if predicted is empty
        - Otherwise standard F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    g_len = len(ground_truth_ids)
    p_len = len(predicted_ids)

    # Singleton case
    if g_len == 0:
        return 1.0 if p_len == 0 else 0.0

    # Non-singleton case but no predictions made
    if p_len == 0:
        return 0.0

    # True positives: intersection of predicted and true matches
    tp = len(predicted_ids.intersection(ground_truth_ids))
    if tp == 0:
        return 0.0

    precision = tp / p_len
    recall = tp / g_len

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return float(f05)


def evaluate_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Computes macro-averaged F_0.5 across all Source 1 entities in ground_truth.

    Args:
        predictions: Dict mapping source1_entity_id -> set of predicted matching entity_ids.
        ground_truth: Dict mapping source1_entity_id -> set of true matching entity_ids.

    Returns:
        Dictionary containing:
            - macro_f05: The primary competition score.
            - singleton_accuracy: Accuracy on singletons.
            - non_singleton_f05: Mean F_0.5 on entities with matches.
            - total_entities: Total S1 entities evaluated.
            - singleton_count: Number of singletons.
    """
    scores = []
    singleton_scores = []
    matched_entity_scores = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_matches, true_matches)
        scores.append(score)

        if len(true_matches) == 0:
            singleton_scores.append(score)
        else:
            matched_entity_scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    non_singleton_f05 = float(np.mean(matched_entity_scores)) if matched_entity_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "total_entities": len(scores),
        "singleton_count": len(singleton_scores),
        "matched_count": len(matched_entity_scores),
    }


def optimize_threshold(
    s1_ids: List[str],
    candidate_ids: List[str],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: np.ndarray = np.arange(0.40, 0.96, 0.02),
) -> Tuple[float, float]:
    """Grid searches the optimal probability threshold that maximizes macro F_0.5.

    Args:
        s1_ids: List of Source 1 entity IDs for each candidate pair.
        candidate_ids: List of candidate entity IDs for each pair.
        probabilities: Model predicted match probabilities for each pair.
        ground_truth: Ground truth mapping for evaluation.
        thresholds: Range of thresholds to evaluate.

    Returns:
        (best_threshold, best_f05_score)
    """
    # Group predictions by s1_id for quick candidate evaluation
    from collections import defaultdict

    pair_data = defaultdict(list)
    for s1, cand, prob in zip(s1_ids, candidate_ids, probabilities):
        pair_data[s1].append((cand, prob))

    best_thresh = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds = {s1: set() for s1 in ground_truth.keys()}
        for s1, candidates in pair_data.items():
            if s1 in preds:
                for cand, p in candidates:
                    if p >= thresh:
                        preds[s1].add(cand)

        res = evaluate_macro_f05(preds, ground_truth)
        score = res["macro_f05"]
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score
