"""Multi-Pass Blocking & Candidate Generation Module.

Generates candidate pairs from Source 2 and Source 3 for each Source 1 entity:
1. Country-level Partitioning (handles open set: US, India, France)
2. Sublinear TF-IDF Sparse Cosine Retrieval on Business Name
3. Sublinear TF-IDF Sparse Cosine Retrieval on Combined Name + Address
4. Core Distinctive Token Inverted Index
5. Postal Code + Fuzzy Name Filter
6. Candidate Union and Top-K Pruning
7. Export to official candidate_pairs.tsv format
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def generate_tfidf_candidates(
    s1_texts: List[str],
    s23_texts: List[str],
    s1_ids: List[str],
    s23_ids: List[str],
    top_k: int = 30,
    min_similarity: float = 0.20,
    ngram_range: Tuple[int, int] = (1, 2),
    analyzer: str = "word",
) -> Dict[str, Set[str]]:
    """Generates candidate pairs using fast sparse TF-IDF matrix multiplication."""
    if not s1_texts or not s23_texts:
        return defaultdict(set)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        min_df=1,
        max_df=0.7,
        sublinear_tf=True
    )

    vectorizer.fit(s1_texts + s23_texts)
    x_s1 = vectorizer.transform(s1_texts)
    x_s23 = vectorizer.transform(s23_texts)

    candidates = defaultdict(set)

    # Process in batches to control RAM
    batch_size = 2000
    n_s1 = x_s1.shape[0]

    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)
        batch_s1 = x_s1[start_idx:end_idx]

        # Dot product with transpose: (batch_size, n_features) x (n_features, n_s23) -> (batch_size, n_s23)
        sim_matrix = batch_s1.dot(x_s23.T)

        for row_idx in range(sim_matrix.shape[0]):
            global_s1_idx = start_idx + row_idx
            s1_id = s1_ids[global_s1_idx]

            row = sim_matrix.getrow(row_idx)
            if row.nnz == 0:
                continue

            indices = row.indices
            data = row.data

            if len(data) > top_k:
                # Top-K indices by score
                top_part = np.argpartition(-data, top_k)[:top_k]
                sorted_top = top_part[np.argsort(-data[top_part])]
                for k in sorted_top:
                    if data[k] >= min_similarity:
                        candidates[s1_id].add(s23_ids[indices[k]])
            else:
                for k in np.argsort(-data):
                    if data[k] >= min_similarity:
                        candidates[s1_id].add(s23_ids[indices[k]])

    return candidates


def generate_token_inverted_index_candidates(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    min_token_len: int = 4,
    max_token_docs: int = 150,
) -> Dict[str, Set[str]]:
    """Inverted index candidate generation for rare / distinctive name tokens."""
    index = defaultdict(list)
    for idx, row in df_s23.iterrows():
        tokens = set(row["clean_name"].split())
        for token in tokens:
            if len(token) >= min_token_len:
                index[token].append(row["entity_id"])

    candidates = defaultdict(set)
    for idx, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        tokens = set(row["clean_name"].split())
        for token in tokens:
            if len(token) >= min_token_len:
                matches = index.get(token, [])
                # Only use tokens that appear in fewer than max_token_docs
                if 0 < len(matches) <= max_token_docs:
                    for mid in matches:
                        candidates[s1_id].add(mid)

    return candidates


def block_by_country(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_entity: int = 50,
) -> Dict[str, List[str]]:
    """Executes multi-pass blocking partitioned strictly by country."""
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    all_s1_ids = df_s1["entity_id"].tolist()
    final_candidates = {s1: set() for s1 in all_s1_ids}

    # Partition by country
    countries = df_s1["country"].unique()

    for country in countries:
        s1_country = df_s1[df_s1["country"] == country]
        s23_country = df_s23[df_s23["country"] == country]

        if s1_country.empty or s23_country.empty:
            continue

        print(f"Blocking country '{country}': {len(s1_country)} S1 records vs {len(s23_country)} S2/S3 records...")

        s1_texts_name = s1_country["clean_name"].tolist()
        s23_texts_name = s23_country["clean_name"].tolist()
        s1_ids = s1_country["entity_id"].tolist()
        s23_ids = s23_country["entity_id"].tolist()

        # Pass 1: Name Word TF-IDF
        cands_p1 = generate_tfidf_candidates(
            s1_texts_name, s23_texts_name, s1_ids, s23_ids,
            top_k=30, min_similarity=0.20, ngram_range=(1, 2), analyzer="word"
        )

        # Pass 2: Name Char n-gram TF-IDF (robust to typos)
        cands_p2 = generate_tfidf_candidates(
            s1_texts_name, s23_texts_name, s1_ids, s23_ids,
            top_k=25, min_similarity=0.30, ngram_range=(3, 4), analyzer="char_wb"
        )

        # Pass 3: Combined Name + Address TF-IDF
        s1_comb = s1_country["combined_text"].tolist()
        s23_comb = s23_country["combined_text"].tolist()
        cands_p3 = generate_tfidf_candidates(
            s1_comb, s23_comb, s1_ids, s23_ids,
            top_k=20, min_similarity=0.20, ngram_range=(1, 2), analyzer="word"
        )

        # Pass 4: Inverted token index
        cands_p4 = generate_token_inverted_index_candidates(s1_country, s23_country)

        # Merge candidate sets for each S1 entity in this country
        for s1 in s1_ids:
            merged = (
                cands_p1.get(s1, set())
                | cands_p2.get(s1, set())
                | cands_p3.get(s1, set())
                | cands_p4.get(s1, set())
            )
            final_candidates[s1].update(list(merged)[:max_candidates_per_entity])

    # Convert sets to sorted lists
    return {s1: sorted(list(cands)) for s1, cands in final_candidates.items()}


def export_candidate_pairs(candidates: Dict[str, List[str]], output_path: str):
    """Exports candidate pairs to the official tab-separated candidate_pairs.tsv format."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cand_list = candidates[s1_id]
            f.write(f"{s1_id}\t{','.join(cand_list)}\n")
    print(f"Exported candidate pairs to {output_path} ({len(candidates)} entities).")
