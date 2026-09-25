# Amazon ML Challenge 2026: Business Entity Resolution

End-to-End Machine Learning Solution for cross-source business entity resolution across noisy records from **Source 1**, **Source 2**, and **Source 3**.

---

## 1. Directory Structure

```text
code/business_entity_resolution/
├── run.py                 # Turnkey pipeline runner and validation trigger
├── requirements.txt       # Minimal, pinned dependencies
├── README.md              # Pipeline documentation and usage instructions
└── src/
    ├── preprocess.py      # Unicode NFKD, legal suffix stripping, address parsing
    ├── blocking.py        # Country-partitioned multi-pass candidate blocking
    ├── features.py        # Discriminative string, token, digit, and interaction features
    ├── model.py           # HistGradientBoostingClassifier & GroupKFold CV
    ├── evaluate.py        # Official macro F_0.5 score & threshold optimization
    └── pipeline.py        # End-to-end orchestration
```

---

## 2. Key Architecture & Methodology

1. **Preprocessing & Cross-Lingual Normalization (`preprocess.py`):**
   - Applies Unicode NFKD decomposition to strip accents/diacritics for cross-lingual robustness (`US`, `India`, `France`).
   - Normalizes corporate legal entity suffixes across regions (`inc`, `llc`, `pvt ltd`, `sarl`, `sas`, etc.) to isolate canonical business root names.
   - Standardizes street, road, unit abbreviations, and extracts numeric/postal tokens.

2. **Partitioned Candidate Blocking (`blocking.py`):**
   - Dynamically partitions by `country` (open set handling) to restrict candidate space to identical jurisdictions ($O(N)$ efficiency).
   - Generates candidate pairs through 4 complementary passes:
     - Word-level TF-IDF sparse cosine similarity on business names
     - Character 3/4-gram TF-IDF sparse cosine similarity (resilient to typos and misspellings)
     - Word-level TF-IDF on combined business name + address text
     - Inverted index on core rare tokens
   - Unions and ranks candidates up to top-$K$ (default: 50) per Source 1 entity.

3. **Pairwise Feature Engineering (`features.py`):**
   - **Name Similarity:** Jaccard (word & character 3-grams), Dice, SequenceMatcher, token sort ratio, root name exact/stem match, first word match.
   - **Address Similarity:** Jaccard, Dice, SequenceMatcher, token sort ratio, postal code exact match.
   - **Token Overlap:** Digits Jaccard similarity, word count differences, source indicator flags.
   - **Non-linear Interactions:** Multiplicative interaction score between name and address similarities.

4. **Model Training & F_0.5 Threshold Tuning (`model.py` & `evaluate.py`):**
   - Multi-threaded `HistGradientBoostingClassifier` natively optimized for fast CPU inference.
   - 5-Fold `GroupKFold` split grouped by `source1_entity_id` to prevent data leakage.
   - Grid search optimization for threshold $\tau^*$ directly maximizing the competition metric:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
     with full singleton support (singletons with zero matches receive $1.0$ if predicted empty).

---

## 3. Installation & Requirements

Ensure Python 3.8+ is installed:

```bash
pip install -r requirements.txt
```

Only standard open-source libraries (`numpy`, `pandas`, `scipy`, `scikit-learn`) are required.

---

## 4. How to Run

### Turnkey Execution:
Run the complete pipeline from the project root:

```bash
python code/business_entity_resolution/run.py \
    --train-dir student_resource/dataset/train \
    --test-dir student_resource/dataset/test \
    --output-dir output
```

### Options:
- `--sample-size N`: Run a fast benchmark on $N$ Source 1 entities (e.g. `--sample-size 5000`).
- `--max-candidates K`: Control maximum candidates generated per entity (default: 50).
- `--skip-validation`: Skip the post-run validation check.

---

## 5. Expected Output

The pipeline generates two tab-separated files inside `--output-dir` (default: `output/`):

1. **`output/matching_results.tsv`** *(Official Leaderboard Submission)*
   - Format: `source1_entity_id\tmatched_entity_ids`
   - Contains **every single** Source 1 entity from `test_source1.tsv`.
   - Matching entities from `Source 2` and `Source 3` are comma-separated (e.g., `S1-100\tS2-4521,S3-9821`).
   - Singletons (no match found above threshold $\tau^*$) have empty second column (e.g., `S1-101\t`).

2. **`output/candidate_pairs.tsv`** *(Blocking Candidates)*
   - Format: `source1_entity_id\tcandidate_entity_ids`
   - Contains candidate pairs generated during the blocking stage.
   - Every entity in `matching_results.tsv` is guaranteed to be a strict subset of `candidate_pairs.tsv`.

3. **Validation Summary:**
   - Automatically runs `student_resource/utils/validate_submission.py` to verify schema, delimiters, column headers, and singleton completeness, ensuring safe submission with exit code 0.
