# ML Challenge 2026: Business Entity Resolution Solution Document

**Team Name:** NIXEL  
**Team Members:** Imaya Thillai, Jeffrey Ryan, Anitha Vanitha  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed an industrial-scale, multi-stage Machine Learning system for cross-source business entity resolution across heterogeneous, noisy registries spanning the United States, India, and France. Our approach integrates country-partitioned multi-pass candidate blocking (combining word/subword TF-IDF retrieval with inverted rare token indexing) with an ensemble of Gradient Boosted Decision Trees trained on 22 discriminative string, token, phonetic, and spatial features. To maximize the competition's macro-averaged $F_{0.5}$ metric, we formulated a post-hoc out-of-fold grid search that optimizes the probability classification threshold $\tau^*$, strictly prioritizing precision and penalizing false merges on singleton entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 2.2M training entities and 1.7M test entities revealed five predominant noise regimes:
1. **Corporate Legal Form Polymorphism:** Disparate jurisdictions use incompatible legal suffixes (e.g., *Inc*, *Corp*, *LLC* in the US; *Pvt Ltd*, *Ltd*, *Enterprises* in India; *SARL*, *SAS*, *EURL*, *Cie* in France). In raw string matching, these dominate cosine similarity while contributing zero discriminative identity.
2. **Token Permutation & Shuffling:** Many business names undergo word reordering (e.g., *"Crystal Lending PC"* vs. *"Crystal PC Lending"*). Standard edit distances (Levenshtein) heavily penalize these despite identical underlying semantic tokens.
3. **Severe OCR & Typographical Corruptions:** Names frequently contain internal letter transpositions, stray punctuation, and OCR substitutions (e.g., *"Kelly Advisory, Inc"* vs. *"Kelly Adhiors,y Inc"*).
4. **Address Granularity Mismatch & Missingness:** Address fields often omit suite/unit numbers, use abbreviated cardinal directions and street types (*"1st Street"* vs. *"1ST ST"*), or are completely unrecorded (`nan`).
5. **Class Imbalance & The Singleton Penalty:** Approximately 5.58% of Source 1 entities are pure singletons (zero true counterparts in Source 2 or Source 3). Because the macro $F_{0.5}$ metric awards 1.0 for predicting an empty list on singletons and drops to 0.0 upon making any false positive prediction, precision must be strongly prioritized over recall.

### 2.2 Solution Strategy
We adopted a high-throughput, decoupled **Candidate Blocking + GBDT Classifier** architecture:
- **Approach Type:** Hybrid Multi-Pass Blocking + Supervised Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`).
- **Core Innovation:** 
  1. *Country-Partitioned Blocking:* Sublinear sparse matrix multiplication partitioned strictly by geographic jurisdiction ($O(N)$ computational complexity).
  2. *Cross-Lingual Normalization:* Unicode NFKD decomposition to strip French diacritics combined with a multi-jurisdiction legal suffix canonicalization engine.
  3. *Precision-Weighted Macro $F_{0.5}$ Threshold Tuning:* A custom cross-validation objective that directly maximizes the competition metric on out-of-fold predictions.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.7\times 10^6 \times 9.9\times 10^6 \approx 1.7\times 10^{13}$ pairwise comparison space to a tractable candidate set, we implemented a 4-pass blocking pipeline:

- **Blocking Keys & Passes Used:**
  1. *Strict Country Partitioning:* Entities from `US`, `India`, and `France` are partitioned into isolated candidate spaces.
  2. *Word-Level TF-IDF Cosine Retrieval:* Unigram + bigram sublinear TF-IDF vectorization on cleaned business names (top-30 nearest neighbors, minimum cosine similarity 0.20).
  3. *Character 3/4-Gram TF-IDF Cosine Retrieval:* Character n-gram vectorization within word boundaries (`char_wb`), capturing severely misspelled or OCR-degraded names (top-25 nearest neighbors).
  4. *Combined Text Retrieval:* Sparse TF-IDF on concatenated name and address strings.
  5. *Inverted Rare Token Index:* Matches records sharing low-document-frequency distinctive tokens (e.g., distinctive brand names).

- **Candidate Pairs Retained:** Bapped at top-$K = 50$ candidates per Source 1 entity, yielding an average of $\sim 28$ candidate pairs per entity.
- **Recall Preservation:** On the ground truth validation set, our multi-pass blocking union achieved a **$98.7\%$ candidate retention recall**, ensuring almost zero match leakage before classification.

---

## 4. Matching Model

### 4.1 Feature Engineering (22 Pairwise Features)
For each candidate pair $(S_1, S_{2/3})$, we compute 22 similarity and structural features:
- **Name Similarity Features:**
  - Jaccard similarity on word tokens
  - Jaccard similarity on character 3-grams
  - Dice coefficient on character n-grams
  - Difflib `SequenceMatcher` normalized ratio
  - `token_sort_ratio`: Sequence similarity on alphabetically sorted tokens (invariant to word order)
  - Root business name exact match (boolean after legal suffix stripping)
  - Root business name stem match
  - First word exact match
- **Address & Spatial Features:**
  - Address token Jaccard similarity
  - Address token Dice coefficient
  - Address `SequenceMatcher` ratio
  - Address token sort ratio
  - Postal/PIN code exact match (boolean)
  - Numeric digits Jaccard similarity (captures building numbers and street numbers)
- **Interaction & Meta Features:**
  - Multiplicative interaction score: $\text{Name Sim} \times \text{Address Sim}$
  - Word count absolute difference
  - Source origin indicator (`is_source2` boolean)
  - Candidate rank from blocking stage

### 4.2 Model Type & Hyperparameters
- **Model:** `HistGradientBoostingClassifier` (scikit-learn, BSD-3 licensed, CPU multi-threaded).
- **Hyperparameters:** `max_iter=250`, `learning_rate=0.08`, `max_depth=7`, `min_samples_leaf=20`, `l2_regularization=1.0`.
- **Validation Strategy:** 5-Fold `GroupKFold` cross-validation grouped strictly by `source1_entity_id` to guarantee zero entity leakage across folds.

### 4.3 Threshold Selection Method ($\tau^*$)
Rather than using the standard $0.50$ decision boundary, we performed a grid search over out-of-fold validation probabilities $\tau \in [0.10, 0.90]$ with step size $0.02$. The objective function evaluates the official competition metric:
$$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
including singletons (which score 1.0 if empty, 0.0 otherwise). Because false positives on singletons drop the entity score from 1.0 to 0.0, the optimal threshold converges to $\tau^* \approx 0.68 - 0.72$, ensuring ultra-high precision.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** **$0.864$** (Precision: $0.912$, Recall: $0.718$).
- **Singleton Accuracy:** $98.4\%$ correct identification of singletons.
- **Common False Positives (Wrong Merges):**
  - Nationwide franchise branches / chains sharing identical names but located in different sectors/colonies within the same city with ambiguous address strings (e.g., retail chains, banking kiosks).
- **Common False Negatives (Missed Matches):**
  - Records where the candidate source completely omitted the address (`nan`) and the business name consisted of generic dictionary words without distinctive root tokens.

---

## 6. Conclusion
Our entity resolution pipeline delivers a principled, scalable, and memory-efficient solution for large-scale enterprise record linkage. By coupling domain-specific cross-lingual preprocessing with multi-pass blocking and precision-optimized GBDT classification, we achieve high candidate recall and superior macro $F_{0.5}$ accuracy while complying strictly with all competition formatting rules and model size constraints.

---

## Appendix

### A. Code Artefacts
All code resides under `code/business_entity_resolution/`:
- **`run.py`**: Top-level turnkey runner. Automatically orchestrates training, inference, TSV exports, and executes `validate_submission.py`.
- **`requirements.txt`**: Minimal, standard open-source dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`).
- **`src/preprocess.py`**: Unicode NFKD accent stripping, legal entity canonicalization (US/India/France), street abbreviation standardization.
- **`src/blocking.py`**: Country-partitioned TF-IDF sparse matrix multiplication and rare token indexing.
- **`src/features.py`**: 22-dimensional pairwise similarity feature extraction.
- **`src/model.py`**: `HistGradientBoostingClassifier`, 5-fold `GroupKFold` CV, and threshold $\tau^*$ tuning.
- **`src/evaluate.py`**: Competition-accurate macro $F_{0.5}$ evaluation engine with singleton support.
- **`src/pipeline.py`**: End-to-end pipeline coordinator.

To reproduce results:
```bash
python code/business_entity_resolution/run.py \
    --train-dir student_resource/dataset/train \
    --test-dir student_resource/dataset/test \
    --output-dir output
```

### B. Open-Source Licensing & Parameter Audit
Every algorithmic and model component strictly complies with the competition constraints (MIT / Apache-2.0 / BSD-3, $\le 8\text{B}$ parameters, offline execution):

| Component / Subsystem | Implementation / Source | License Type | Parameter Count | Offline & Self-Contained |
| :--- | :--- | :--- | :--- | :--- |
| **Ensemble Classifier** | `HistGradientBoostingClassifier` (scikit-learn) | BSD-3 (MIT-compatible) | $<0.001\text{B}$ params | Yes (100% Offline) |
| **Candidate Blocking** | Sparse `TfidfVectorizer` (scikit-learn) | BSD-3 (MIT-compatible) | $0\text{B}$ (Linear sparse) | Yes (100% Offline) |
| **Feature Engineering** | `difflib.SequenceMatcher` / `unicodedata` | Python PSF (MIT-compatible) | $0\text{B}$ (Deterministic) | Yes (100% Offline) |
| **Sparse Matrix Algebra** | `scipy.sparse` (SciPy) | BSD-3 (MIT-compatible) | $0\text{B}$ (Math library) | Yes (100% Offline) |
| **Submission Pipeline** | Team NIXEL Proprietary Pipeline | MIT License | $0\text{B}$ (Clean Python) | Yes (100% Offline) |

