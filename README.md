# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

End-to-end Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Project Motive

In real-world e-commerce, logistics, and catalog intelligence systems, business entity records are ingested continuously from heterogeneous, unstandardized third-party data providers. Each provider introduces disparate formatting conventions, phonetic spellings, regional abbreviations, legal entity suffixes, and typographical errors.

### The Objective:
Given three distinct datasets:
- **`Source 1` (Reference Entities):** The primary reference database.
- **`Source 2` & `Source 3` (Candidate Sources):** Noisy complementary business records.

For every business entity in `Source 1`, identify and link all corresponding matching records belonging to the same physical commercial establishment in `Source 2` and `Source 3`. The challenge encompasses international business records across multiple legal jurisdictions:
- **US** (United States)
- **India**
- **France**

### The Evaluation Metric:
Submissions are evaluated on **Macro-Averaged $F_{0.5}$ Score** across all `Source 1` entities:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R}$$

- **Precision is weighted $2\times$ as heavily as Recall**: False positives severely degrade search trust and catalog accuracy.
- **Singletons (Entities with 0 true matches)**: If an entity has no match in Source 2/3, predicting an empty list scores **1.0**; predicting any false match drops its score to **0.0**.

---

## 2. Pipeline Architecture

```text
                               +----------------------------------+
                               | Raw Sources: S1, S2, and S3      |
                               +----------------------------------+
                                                 |
                                                 v
                       +--------------------------------------------------+
                       | Stage 1: Cross-Lingual Preprocessing             |
                       |  - Unicode NFKD Accent Stripping (French/US/IN)  |
                       |  - Legal Suffix Normalization (SARL, Pvt Ltd...) |
                       |  - Address Standardization & Postal Extraction   |
                       +--------------------------------------------------+
                                                 |
                                                 v
                       +--------------------------------------------------+
                       | Stage 2: Country-Partitioned Multi-Pass Blocking |
                       |  - Strict Country Partitioning (O(N) scaling)    |
                       |  - Word & Char n-gram Sparse TF-IDF Cosine Sim   |
                       |  - Inverted Rare Token Index                     |
                       +--------------------------------------------------+
                                                 |
                                                 v
                       +--------------------------------------------------+
                       | Stage 3: Pairwise Feature Engineering            |
                       |  - 22 Features: Jaccard, Dice, SequenceMatcher,  |
                       |    Token Sort, Root Name Match, Digits, Postal   |
                       +--------------------------------------------------+
                                                 |
                                                 v
                       +--------------------------------------------------+
                       | Stage 4: HistGradientBoosting & CV Optimization  |
                       |  - 5-Fold GroupKFold CV (grouped by S1 ID)       |
                       |  - Precision-Heavy Optimal Threshold (tau*) Grid |
                       +--------------------------------------------------+
                                                 |
                                                 v
                       +--------------------------------------------------+
                       | Stage 5: Inference & Submission Generation       |
                       |  - output/matching_results.tsv (Leaderboard)     |
                       |  - output/candidate_pairs.tsv  (Blocking pairs)  |
                       |  - utils/validate_submission.py Check (PASS)     |
                       +--------------------------------------------------+
```

---

## 3. Official Submission Deliverables

Per the official competition guidelines, the final submission requires:

1. **`matching_results.tsv`** *(Mandatory — The Scored File)*:
   - Evaluated on the leaderboard.
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Must contain every single Source 1 entity from `test_source1.tsv`.
   - Singletons must have an empty matched column (e.g., `S1-001\t`).
2. **`candidate_pairs.tsv`** *(Mandatory in Zip)*:
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Candidate pairs generated by blocking; matched IDs must be a strict subset of candidate IDs.
3. **`code/business_entity_resolution/`**:
   - `src/` modular source files (`preprocess.py`, `blocking.py`, `features.py`, `model.py`, `evaluate.py`, `pipeline.py`).
   - `requirements.txt` with pinned dependencies.
   - `README.md` with instructions.
4. **`Documentation_template.md`**:
   - Technical methodology write-up explaining candidate generation recall, feature engineering, and cross-validation scores.
5. **Submission Zip Package**:
   - Zipped archive `<team_name>_submission.zip` matching the required directory tree.

---

## 4. Quick Start

### Installation:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Run on Benchmark Sample Data:
```bash
python code/business_entity_resolution/run.py \
    --train-dir sample_data/train \
    --test-dir sample_data/test \
    --output-dir sample_output
```

### Run on Full Competition Dataset:
```bash
python code/business_entity_resolution/run.py \
    --train-dir student_resource/dataset/train \
    --test-dir student_resource/dataset/test \
    --output-dir output
```

### Submission Validation:
Every run automatically invokes `student_resource/utils/validate_submission.py`. You can also run it independently:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

---

## 5. Repository Structure

```text
.
├── code/
│   └── business_entity_resolution/
│       ├── run.py                 # Pipeline entry point & validation
│       ├── requirements.txt       # Minimal dependencies (numpy, pandas, scikit-learn, scipy)
│       ├── README.md              # Code execution documentation
│       └── src/
│           ├── preprocess.py      # Unicode NFKD, legal suffixes, address parsing
│           ├── blocking.py        # Multi-pass sparse TF-IDF & country partitioning
│           ├── features.py        # 22 pairwise similarity & interaction features
│           ├── model.py           # HistGradientBoosting & GroupKFold CV
│           ├── evaluate.py        # Macro F_0.5 metric & grid search optimizer
│           └── pipeline.py        # Complete pipeline orchestration
├── sample_data/                   # Turnkey verification benchmark data
├── student_resource/
│   ├── Documentation_template.md  # Official methodology document template
│   ├── README.md                  # Official challenge rules & format specification
│   └── utils/
│       └── validate_submission.py # Official stdlib submission validator
└── .gitignore                     # Excludes large TSV files & binaries (>100MB)
```
