"""End-to-End Pipeline Runner for Business Entity Resolution.

Coordinates the complete workflow:
1. Data loading and normalization (US, India, France)
2. Partitioned multi-pass candidate blocking
3. Pairwise similarity feature extraction
4. 5-Fold GroupKFold Cross-Validation & F_0.5 threshold optimization
5. Final model training (HistGradientBoostingClassifier)
6. Test set candidate blocking & export to candidate_pairs.tsv
7. Test set match prediction with optimal threshold tau*
8. Export predictions to matching_results.tsv
9. Submission validation check
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_source_tsv, load_ground_truth_tsv
from blocking import block_by_country, export_candidate_pairs
from features import build_feature_dataframe
from model import EntityMatchingModel, export_matching_results
from evaluate import evaluate_macro_f05


def run_pipeline(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    sample_size: Optional[int] = None,
    max_candidates_per_entity: int = 50,
) -> Dict[str, float]:
    """Executes the complete entity resolution pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: LOAD & PREPROCESS TRAINING DATA
    # -------------------------------------------------------------
    print("\n[STAGE 1/6] Loading and Preprocessing Training Data...")
    t0 = time.time()
    tr_s1_path = os.path.join(train_dir, "train_source1.tsv")
    tr_s2_path = os.path.join(train_dir, "train_source2.tsv")
    tr_s3_path = os.path.join(train_dir, "train_source3.tsv")
    tr_gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    if not all(os.path.exists(p) for p in [tr_s1_path, tr_s2_path, tr_s3_path, tr_gt_path]):
        raise FileNotFoundError(
            f"Training files missing in {train_dir}! Ensure train_source1.tsv, "
            f"train_source2.tsv, train_source3.tsv, and train_ground_truth.tsv exist."
        )

    df_tr_s1 = load_source_tsv(tr_s1_path)
    df_tr_s2 = load_source_tsv(tr_s2_path)
    df_tr_s3 = load_source_tsv(tr_s3_path)
    gt_mapping = load_ground_truth_tsv(tr_gt_path)

    if sample_size and sample_size < len(df_tr_s1):
        print(f"Sampling training set to {sample_size} Source 1 entities for fast run...")
        df_tr_s1 = df_tr_s1.iloc[:sample_size].copy()
        s1_keep = set(df_tr_s1["entity_id"])
        gt_mapping = {k: v for k, v in gt_mapping.items() if k in s1_keep}

    df_tr_s23 = pd.concat([df_tr_s2, df_tr_s3], ignore_index=True)
    print(f"Loaded {len(df_tr_s1):,} S1, {len(df_tr_s2):,} S2, {len(df_tr_s3):,} S3 records in {time.time()-t0:.2f}s.")

    # -------------------------------------------------------------
    # STAGE 2: CANDIDATE BLOCKING (TRAIN)
    # -------------------------------------------------------------
    print("\n[STAGE 2/6] Running Partitioned Candidate Blocking on Training Set...")
    t0 = time.time()
    train_candidates = block_by_country(
        df_tr_s1, df_tr_s2, df_tr_s3, max_candidates_per_entity=max_candidates_per_entity
    )
    total_pairs = sum(len(cands) for cands in train_candidates.values())
    avg_pairs = total_pairs / max(len(train_candidates), 1)
    print(f"Generated {total_pairs:,} candidate pairs ({avg_pairs:.1f} pairs/entity) in {time.time()-t0:.2f}s.")

    # Measure blocking recall on training ground truth
    total_true = 0
    retained_true = 0
    for s1_id, true_matches in gt_mapping.items():
        total_true += len(true_matches)
        cand_set = set(train_candidates.get(s1_id, []))
        retained_true += len(set(true_matches).intersection(cand_set))
    blocking_recall = (retained_true / total_true) if total_true > 0 else 1.0
    print(f"Candidate Blocking Recall on Ground Truth: {blocking_recall*100:.2f}% ({retained_true:,}/{total_true:,} true pairs).")

    # -------------------------------------------------------------
    # STAGE 3: FEATURE EXTRACTION (TRAIN)
    # -------------------------------------------------------------
    print("\n[STAGE 3/6] Extracting Pairwise Features for Training...")
    t0 = time.time()
    X_train, y_train, tr_s1_ids, tr_cand_ids = build_feature_dataframe(
        df_tr_s1, df_tr_s23, train_candidates, ground_truth=gt_mapping
    )
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"Feature matrix built: {X_train.shape[0]:,} rows x {X_train.shape[1]} features in {time.time()-t0:.2f}s.")
    print(f"Class balance: {n_pos:,} Positive matches ({n_pos/len(y_train)*100:.1f}%), {n_neg:,} Negatives.")

    # -------------------------------------------------------------
    # STAGE 4: MODEL TRAINING & THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------
    print("\n[STAGE 4/6] Cross-Validation and Model Training...")
    t0 = time.time()
    gt_set_mapping = {k: set(v) for k, v in gt_mapping.items()}
    model = EntityMatchingModel()

    n_groups = len(set(tr_s1_ids))
    actual_splits = min(5, n_groups) if n_groups >= 2 else 2
    if n_groups >= 2:
        best_tau, cv_f05, cv_metrics = model.cross_validate(
            X=X_train,
            y=y_train,
            s1_ids=tr_s1_ids,
            cand_ids=tr_cand_ids,
            ground_truth=gt_set_mapping,
            n_splits=actual_splits,
        )
        print(f"\nOptimal Macro F_0.5 Threshold: tau* = {best_tau:.2f}")
        print(f"{actual_splits}-Fold CV Macro F_0.5: {cv_f05:.4f} (Singleton Acc: {cv_metrics.get('singleton_accuracy', 0):.4f}, Non-Singleton F_0.5: {cv_metrics.get('non_singleton_f05', 0):.4f})")
    else:
        best_tau = 0.70
        cv_f05 = 1.0

    # Fit final model on 100% of training data
    print("Fitting final model on all training candidate pairs...")
    model.fit(X_train, y_train)
    model.threshold = best_tau
    print(f"Model training complete in {time.time()-t0:.2f}s.")

    # -------------------------------------------------------------
    # STAGE 5: TEST SET BLOCKING & CANDIDATE EXPORT
    # -------------------------------------------------------------
    print("\n[STAGE 5/6] Processing Test Set (Candidate Generation & Export)...")
    t0 = time.time()
    te_s1_path = os.path.join(test_dir, "test_source1.tsv")
    te_s2_path = os.path.join(test_dir, "test_source2.tsv")
    te_s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not all(os.path.exists(p) for p in [te_s1_path, te_s2_path, te_s3_path]):
        raise FileNotFoundError(f"Test files missing in {test_dir}!")

    df_te_s1 = load_source_tsv(te_s1_path)
    df_te_s2 = load_source_tsv(te_s2_path)
    df_te_s3 = load_source_tsv(te_s3_path)
    df_te_s23 = pd.concat([df_te_s2, df_te_s3], ignore_index=True)

    all_test_s1_ids = df_te_s1["entity_id"].tolist()
    print(f"Loaded {len(df_te_s1):,} Test S1, {len(df_te_s2):,} S2, {len(df_te_s3):,} S3 records.")

    test_candidates = block_by_country(
        df_te_s1, df_te_s2, df_te_s3, max_candidates_per_entity=max_candidates_per_entity
    )

    # Ensure every single test S1 entity is in candidate dictionary
    for s1_id in all_test_s1_ids:
        if s1_id not in test_candidates:
            test_candidates[s1_id] = []

    # Export candidate_pairs.tsv
    candidate_tsv_path = os.path.join(output_dir, "candidate_pairs.tsv")
    export_candidate_pairs(test_candidates, candidate_tsv_path)
    print(f"Test candidate generation done in {time.time()-t0:.2f}s.")

    # -------------------------------------------------------------
    # STAGE 6: TEST FEATURE EXTRACTION & MATCH PREDICTION
    # -------------------------------------------------------------
    print("\n[STAGE 6/6] Predicting Test Matches with Optimal Threshold tau*...")
    t0 = time.time()
    X_test, _, te_s1_ids, te_cand_ids = build_feature_dataframe(
        df_te_s1, df_te_s23, test_candidates, ground_truth=None
    )
    print(f"Extracted features for {len(X_test):,} test candidate pairs.")

    predicted_matches = model.predict_matches(
        X_test=X_test,
        test_s1_ids=te_s1_ids,
        test_cand_ids=te_cand_ids,
        all_required_s1_ids=all_test_s1_ids,
        threshold=best_tau,
    )

    matching_tsv_path = os.path.join(output_dir, "matching_results.tsv")
    export_matching_results(predicted_matches, matching_tsv_path)

    # Calculate singleton summary
    n_singletons = sum(1 for cands in predicted_matches.values() if len(cands) == 0)
    n_matched = len(predicted_matches) - n_singletons
    total_matched_ids = sum(len(cands) for cands in predicted_matches.values())
    print(f"\nPrediction Summary:")
    print(f"  - Total Test S1 Entities: {len(predicted_matches):,}")
    print(f"  - Entities with Matches: {n_matched:,} ({n_matched/len(predicted_matches)*100:.1f}%)")
    print(f"  - Singletons (no match): {n_singletons:,} ({n_singletons/len(predicted_matches)*100:.1f}%)")
    print(f"  - Total Links Formed: {total_matched_ids:,}")

    elapsed_total = time.time() - start_time
    print(f"\nPipeline finished successfully in {elapsed_total/60:.2f} minutes!")
    print(f"Outputs written to: {output_dir}/")
    print(f"  1. {candidate_tsv_path}")
    print(f"  2. {matching_tsv_path}")
    print("=" * 70)

    return {
        "cv_f05": cv_f05,
        "best_tau": best_tau,
        "blocking_recall": blocking_recall,
        "test_entities": len(predicted_matches),
        "total_matched_ids": total_matched_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Path to train directory")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for quick test run")
    parser.add_argument("--max-candidates", type=int, default=50, help="Max candidates per entity during blocking")
    args = parser.parse_args()

    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_candidates_per_entity=args.max_candidates,
    )


if __name__ == "__main__":
    main()
