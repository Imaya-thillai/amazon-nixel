#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Main Execution Script.

Runs the complete Business Entity Resolution pipeline:
1. Multi-pass candidate blocking
2. Pairwise feature extraction
3. HistGradientBoostingClassifier training & F_0.5 threshold optimization
4. Candidate generation & match prediction on test set
5. TSV export to matching_results.tsv and candidate_pairs.tsv
6. Automatic validation via validate_submission.py
"""

import argparse
import os
import subprocess
import sys

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import run_pipeline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="student_resource/dataset/train",
        help="Path to training dataset directory containing train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="student_resource/dataset/test",
        help="Path to test dataset directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write output TSV files (matching_results.tsv and candidate_pairs.tsv)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of Source 1 entities to sample for training & inference (default: 5000 for fast, memory-safe execution; pass 0 for full 2.2M dataset)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Max candidate pairs to retain per Source 1 entity during blocking",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the post-run validate_submission.py check",
    )
    args = parser.parse_args()

    # 1. Run Pipeline
    results = run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_candidates_per_entity=args.max_candidates,
    )

    # 2. Run Validation if available
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    validator_path = os.path.join("student_resource", "utils", "validate_submission.py")

    if not args.skip_validation and os.path.exists(validator_path):
        print("\n" + "=" * 70)
        print("  RUNNING OFFICIAL SUBMISSION VALIDATOR")
        print("=" * 70)
        val_cmd = [
            sys.executable,
            validator_path,
            "--matching",
            matching_path,
            "--candidate",
            candidate_path,
            "--test-dir",
            args.test_dir,
        ]
        ret = subprocess.run(val_cmd)
        if ret.returncode == 0:
            print("\n[SUCCESS] Official validation passed! Submission is safe to submit.")
        else:
            print(f"\n[WARNING] Validation exited with return code {ret.returncode}. Check messages above.")


if __name__ == "__main__":
    main()
