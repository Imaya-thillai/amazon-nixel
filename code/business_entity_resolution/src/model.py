"""Entity Matching Classifier & Training Pipeline.

Trains Gradient Boosted Decision Trees on pairwise similarity features,
evaluates out-of-fold macro F_0.5 performance, optimizes threshold tau*,
and generates final matching_results.tsv.
"""

import os
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from evaluate import evaluate_macro_f05, optimize_threshold


class EntityMatchingModel:
    """Supervised entity resolution matching model."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_depth=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self.threshold = 0.70  # Default precision-heavy threshold
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits classifier on training feature matrix."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns match probability P(match=1) for each candidate pair."""
        return self.model.predict_proba(X)[:, 1]

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        s1_ids: List[str],
        cand_ids: List[str],
        ground_truth: Dict[str, Set[str]],
        n_splits: int = 5,
    ) -> Tuple[float, float, Dict]:
        """Performs 5-Fold GroupKFold CV grouped by source1_entity_id."""
        gkf = GroupKFold(n_splits=n_splits)
        oof_probs = np.zeros(len(X))

        print(f"Starting {n_splits}-Fold GroupKFold Cross-Validation...")
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=s1_ids), start=1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            clf = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_depth=7,
                min_samples_leaf=20,
                random_state=self.random_state + fold,
            )
            clf.fit(X_tr, y_tr)
            val_preds = clf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_preds
            print(f"  Fold {fold} complete.")

        print("Optimizing probability threshold on Out-Of-Fold predictions...")
        best_tau, best_score = optimize_threshold(
            s1_ids=s1_ids,
            candidate_ids=cand_ids,
            probabilities=oof_probs,
            ground_truth=ground_truth,
        )
        self.threshold = best_tau

        # Final OOF evaluation
        oof_preds = {s1: set() for s1 in ground_truth.keys()}
        for s1, cand, prob in zip(s1_ids, cand_ids, oof_probs):
            if prob >= self.threshold and s1 in oof_preds:
                oof_preds[s1].add(cand)

        metrics = evaluate_macro_f05(oof_preds, ground_truth)
        print(f"\n=== Validation Summary ===")
        print(f"Optimal Threshold (tau*): {best_tau:.3f}")
        print(f"OOF Macro F_0.5 Score:    {metrics['macro_f05']:.4f}")
        print(f"Singleton Accuracy:       {metrics['singleton_accuracy']:.4f} ({metrics['singleton_count']} singletons)")
        print(f"Non-Singleton F_0.5:      {metrics['non_singleton_f05']:.4f} ({metrics['matched_count']} matched)")

        # Retrain full model on 100% of training data
        print("\nRetraining model on full dataset...")
        self.fit(X, y)

        return best_tau, metrics["macro_f05"], metrics

    def predict_matches(
        self,
        X_test: pd.DataFrame,
        test_s1_ids: List[str],
        test_cand_ids: List[str],
        all_required_s1_ids: List[str],
        threshold: float = None,
    ) -> Dict[str, List[str]]:
        """Generates final predictions for all required test S1 entities."""
        tau = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X_test) if len(X_test) > 0 else np.array([])

        matches = {s1: [] for s1 in all_required_s1_ids}
        for s1, cand, prob in zip(test_s1_ids, test_cand_ids, probs):
            if prob >= tau and s1 in matches:
                matches[s1].append(cand)

        # De-duplicate while preserving order
        clean_matches = {}
        for s1, cands in matches.items():
            seen = set()
            clean_list = []
            for c in cands:
                if c not in seen and (c.startswith("S2-") or c.startswith("S3-")):
                    seen.add(c)
                    clean_list.append(c)
            clean_matches[s1] = clean_list

        return clean_matches


def export_matching_results(
    matches: Dict[str, List[str]],
    output_path: str,
):
    """Exports final predictions to official tab-separated matching_results.tsv."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(matches.keys()):
            id_str = ",".join(matches[s1_id])
            f.write(f"{s1_id}\t{id_str}\n")
    print(f"Exported matching results to {output_path} ({len(matches)} entities).")
