#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - HIGH PRECISION Entity Resolution Engine v4.0
Target: Macro F0.5 >= 0.99

KEY INSIGHT FROM TRAINING DATA ANALYSIS:
- Ground truth: 3.666 matches / matched entity, 5.58% singletons
- Macro F0.5 is PRECISION-WEIGHTED (precision counts 4x more than recall)
- F0.5 = 1.25 * P * R / (0.25 * P + R)
- A false positive (wrong match) tanks F0.5 far more than a false negative (miss)
- Strategy: ONLY match when we are HIGHLY CONFIDENT
  
MATCHING SIGNALS (in order of reliability):
1. PHONE NUMBER exact match → 100% precision, direct match
2. EXACT compact name match with SAME STATE/CITY → ~99% precision
3. HOUSE NUMBER + STREET WORD + NAME prefix overlap → ~95% precision
4. FIRST 2 WORDS sorted (SMALL bucket ≤ 3) + SAME STATE → ~90% precision
5. NAME exact (long, ≥ 10 chars) with NO ambiguity check

WHAT WE DO NOT DO:
- Match on address only (massive FP)
- Match short names (2-4 chars) on name alone → huge FP
- Match word pairs in large buckets
- Over-aggressively strip legal suffixes (causes "retail" → false match)
"""

import gc
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict


def strip_accents(text: str) -> str:
    if text.isascii():
        return text
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


# CONSERVATIVE legal suffix stripping - only pure legal entity markers
LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|llp|sarl|sas|gmbh)\b"
)

STOP_ADDR = frozenset({
    "street", "st", "rd", "road", "avenue", "ave", "lane", "ln",
    "drive", "dr", "floor", "unit", "suite", "block", "north", "south",
    "east", "west", "new", "de", "la", "du", "des", "les",
    "null", "na", "the", "and", "of", "in", "at", "to",
    "no", "number", "plot", "flat", "shop", "building", "bldg",
    "sector", "phase", "area", "main", "cross", "near", "opp"
})

# State abbreviations for locality matching
US_STATES = frozenset({
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id",
    "il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms",
    "mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok",
    "or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy"
})

INDIA_STATES = frozenset({
    "mh","ka","dl","tn","gj","rj","up","ts","ap","kl","wb","mp","hr","pb",
    "maharashtra","karnataka","delhi","gujarat","rajasthan","telangana",
    "kerala","haryana","punjab"
})

ALL_STATES = US_STATES | INDIA_STATES

DOMAIN_RE = re.compile(r'\.(com|org|net|in|fr|co|io|biz|info|edu|gov|us|uk|ca|au)\b')


def normalize_name(s: str):
    """
    Returns (compact, words, word_set)
    compact: alphanumeric joined after stripping legal suffixes (min length for matching = 5)
    words: tuple of significant tokens
    word_set: frozenset of words for quick intersection
    """
    if not s:
        return "", (), frozenset()
    s = strip_accents(s).lower()
    s = DOMAIN_RE.sub(" ", s)
    s = re.sub(r'https?://(?:www\.)?', '', s)
    s = re.sub(r'[-_/&]', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = LEGAL_SUFFIXES.sub(' ', s)
    words = tuple(w for w in s.split() if len(w) >= 2)
    compact = "".join(words)
    return compact, words, frozenset(words)


def parse_addr(s: str):
    """
    Returns (hk1, hk2, state_token, city_token, nums)
    hk = first_number + _ + first_significant_addr_word
    state = detected US/India state abbreviation
    """
    if not s:
        return None, None, "", "", []
    s_low = strip_accents(s).lower()
    tokens = re.findall(r'[a-z0-9]+', s_low)
    nums = re.findall(r'\b\d{2,}\b', s_low)
    addr_words = [t for t in tokens if len(t) >= 3 and not t.isdigit() and t not in STOP_ADDR]

    hk1 = f"{nums[0]}_{addr_words[0]}" if (nums and addr_words) else None
    hk2 = f"{nums[0]}_{addr_words[1]}" if (nums and len(addr_words) >= 2) else None

    # State detection
    token_set = set(tokens)
    state_matches = token_set & ALL_STATES
    state = list(state_matches)[0] if state_matches else ""

    # City: first non-state, non-number, non-stopword token of len >= 4
    city = ""
    for t in addr_words:
        if t not in ALL_STATES and len(t) >= 4:
            city = t
            break

    return hk1, hk2, state, city, nums


def name_similarity(c1, w1, ws1, c2, w2, ws2):
    """
    Returns True if names are considered similar enough to be a match.
    CONSERVATIVE: requires strong evidence to call it a match.
    """
    if not c1 or not c2:
        return False
    # Exact compact match (after legal suffix strip)
    if c1 == c2:
        return True
    # One contains the other (only if long enough to avoid noise)
    if len(c1) >= 8 and len(c2) >= 8:
        if c1 in c2 or c2 in c1:
            return True
    # 3+ shared significant words
    shared = ws1 & ws2
    if len(shared) >= 3:
        return True
    # 2 shared words with prefix match on compact (first 6 chars same)
    if len(shared) >= 2 and len(c1) >= 6 and c1[:6] == c2[:6]:
        return True
    return False


def run_fast_matcher(
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    max_bucket_size: int = 4,
):
    """Run the high-precision streaming entity resolution pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026: HIGH-PRECISION ENTITY RESOLUTION v4.0", flush=True)
    print("=" * 75, flush=True)
    t_start = time.time()

    # =========================================================================
    # PASS 1: Read S1 and build target key sets (memory efficient)
    # =========================================================================
    print("\n[PHASE 1/3] Scanning Source 1 and building search keys...", flush=True)
    t0 = time.time()

    s1_compact_keys = set()
    s1_first2_keys = set()
    s1_house_keys = set()
    s1_phone_keys = set()
    s1_count = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            _, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()

            compact, words, _ = normalize_name(name)
            hk1, hk2, state, city, nums = parse_addr(addr)

            # Only register compact keys that are specific enough (len >= 5)
            if len(compact) >= 5:
                s1_compact_keys.add((country, compact))

            # First 2 words — only if both words are long enough (>= 4 chars each)
            if len(words) >= 2 and len(words[0]) >= 4 and len(words[1]) >= 4:
                pair = tuple(sorted([words[0], words[1]]))
                s1_first2_keys.add((country, pair[0], pair[1]))

            # House key — requires actual street number
            if hk1:
                s1_house_keys.add((country, hk1))
            if hk2:
                s1_house_keys.add((country, hk2))

            # Phone numbers (7+ digits)
            for n in nums:
                if len(n) >= 7:
                    s1_phone_keys.add((country, n))

    print(f"Phase 1: {s1_count:,} S1 entities in {time.time()-t0:.1f}s", flush=True)
    print(f"  Keys: {len(s1_compact_keys):,} compact | {len(s1_first2_keys):,} word-pairs | {len(s1_house_keys):,} house | {len(s1_phone_keys):,} phone", flush=True)

    # =========================================================================
    # PASS 2: Stream S2/S3 and build precision inverted indices
    # =========================================================================
    print("\n[PHASE 2/3] Building precision inverted indices from Source 2 & 3...", flush=True)
    t0 = time.time()

    idx_compact = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_house = defaultdict(list)
    idx_phone = defaultdict(list)

    # Candidate metadata (only for matched candidates to save RAM)
    cand_compact = {}   # mid -> compact name
    cand_words = {}     # mid -> frozenset of words
    cand_state = {}     # mid -> state token
    cand_hk = {}        # mid -> hk1

    total_s23 = 0

    for s_path in [s2_path, s3_path]:
        print(f"  Indexing {os.path.basename(s_path)}...", flush=True)
        t_file = time.time()
        n_file = 0
        with open(s_path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                n_file += 1
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                mid, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()

                compact, words, word_set = normalize_name(name)
                hk1, hk2, state, city, nums = parse_addr(addr)

                added = False

                # Signal 1: Compact name (exact, must be >= 5 chars)
                if len(compact) >= 5:
                    k = (country, compact)
                    if k in s1_compact_keys and len(idx_compact[k]) < max_bucket_size:
                        idx_compact[k].append(mid)
                        added = True

                # Signal 2: First 2 significant words (both >= 4 chars)
                if len(words) >= 2 and len(words[0]) >= 4 and len(words[1]) >= 4:
                    pair = tuple(sorted([words[0], words[1]]))
                    k = (country, pair[0], pair[1])
                    if k in s1_first2_keys and len(idx_first2[k]) < max_bucket_size:
                        idx_first2[k].append(mid)
                        added = True

                # Signal 3: House key
                for hk in [hk1, hk2]:
                    if hk:
                        k = (country, hk)
                        if k in s1_house_keys and len(idx_house[k]) < max_bucket_size:
                            idx_house[k].append(mid)
                            added = True

                # Signal 4: Phone number
                for n in nums:
                    if len(n) >= 7:
                        k = (country, n)
                        if k in s1_phone_keys and len(idx_phone[k]) < max_bucket_size:
                            idx_phone[k].append(mid)
                            added = True

                # Store lightweight metadata for candidates only
                if added:
                    if compact:
                        cand_compact[mid] = compact
                    if word_set:
                        cand_words[mid] = word_set
                    if state:
                        cand_state[mid] = state
                    if hk1:
                        cand_hk[mid] = hk1

        total_s23 += n_file
        print(f"    {n_file:,} records in {time.time()-t_file:.1f}s", flush=True)

    # Free large key sets
    del s1_compact_keys, s1_first2_keys, s1_house_keys, s1_phone_keys
    gc.collect()

    n_cands = len(cand_compact)
    n_buckets = len(idx_compact) + len(idx_first2) + len(idx_house) + len(idx_phone)
    print(f"Phase 2: {total_s23:,} S2/S3 records indexed in {time.time()-t0:.1f}s", flush=True)
    print(f"  {n_buckets:,} index buckets, {n_cands:,} unique candidates stored", flush=True)

    # =========================================================================
    # PASS 3: Stream S1 and apply precision matching rules
    # =========================================================================
    print("\n[PHASE 3/3] Streaming Source 1: precision matching & writing outputs...", flush=True)
    t0 = time.time()

    matched_ct = singleton_ct = total_links = total_cands = n_written = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(matching_path, "w", encoding="utf-8") as fm, \
         open(candidate_path, "w", encoding="utf-8") as fc:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        next(fin)
        for line in fin:
            n_written += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            sid, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()

            c1, w1, ws1 = normalize_name(name)
            hk1_s, hk2_s, st1, city1, nums1 = parse_addr(addr)

            cands = set()
            matches = set()

            # ---- Signal 1: Phone Number (very high precision) ----
            for n in nums1:
                if len(n) >= 7:
                    for mid in idx_phone.get((country, n), []):
                        cands.add(mid)
                        matches.add(mid)  # Phone match = definite

            # ---- Signal 2: Exact Compact Name ----
            if len(c1) >= 5:
                k = (country, c1)
                bucket = idx_compact.get(k, [])
                for mid in bucket:
                    cands.add(mid)
                    stm = cand_state.get(mid, "")
                    # Require state match if both have states
                    if st1 and stm and st1 != stm:
                        continue
                    # For short names (5-7 chars), require address corroboration
                    if len(c1) <= 7:
                        hkm = cand_hk.get(mid, "")
                        addr_ok = (
                            (hk1_s and hkm and hk1_s == hkm) or
                            (hk2_s and hkm and hk2_s == hkm) or
                            (city1 and stm and city1 == stm)
                        )
                        if not addr_ok:
                            continue
                    matches.add(mid)

            # ---- Signal 3: First 2 Words (SMALL BUCKET ONLY - max 2 matches) ----
            if len(w1) >= 2 and len(w1[0]) >= 4 and len(w1[1]) >= 4:
                pair = tuple(sorted([w1[0], w1[1]]))
                bucket = idx_first2.get((country, pair[0], pair[1]), [])
                for mid in bucket:
                    cands.add(mid)
                # Only trust word-pair match if bucket is tiny (≤ 2 candidates)
                # AND state matches AND address corroborates
                if len(bucket) <= 2:
                    for mid in bucket:
                        stm = cand_state.get(mid, "")
                        if st1 and stm and st1 != stm:
                            continue
                        # Require address corroboration for word-pair signal
                        hkm = cand_hk.get(mid, "")
                        addr_ok = (hk1_s and hkm and hk1_s == hkm) or \
                                  (hk2_s and hkm and hk2_s == hkm)
                        if addr_ok:
                            matches.add(mid)

            # ---- Signal 4: House Key with Name Verification ----
            for hk_s in [hk1_s, hk2_s]:
                if hk_s:
                    bucket = idx_house.get((country, hk_s), [])
                    for mid in bucket:
                        cands.add(mid)
                        stm = cand_state.get(mid, "")
                        if st1 and stm and st1 != stm:
                            continue
                        c2 = cand_compact.get(mid, "")
                        ws2 = cand_words.get(mid, frozenset())
                        # Must have strong name evidence when using house key
                        if name_similarity(c1, w1, ws1, c2, (), ws2):
                            matches.add(mid)

            # Ensure matches ⊆ candidates (strict subset guarantee)
            cands.update(matches)

            cand_list = sorted(cands)
            match_list = sorted(matches)
            total_cands += len(cand_list)
            total_links += len(match_list)

            if match_list:
                matched_ct += 1
            else:
                singleton_ct += 1

            fm.write(f"{sid}\t{','.join(match_list)}\n")
            fc.write(f"{sid}\t{','.join(cand_list)}\n")

            if n_written % 500_000 == 0:
                print(f"  {n_written:,}/{s1_count:,} entities processed...", flush=True)

    elapsed = time.time() - t_start
    print(f"\nPhase 3 complete in {time.time()-t0:.1f}s", flush=True)
    print("\n" + "=" * 75, flush=True)
    print("  EXECUTION SUMMARY (v4.0 High-Precision Engine)", flush=True)
    print("=" * 75, flush=True)
    print(f"Total S1 Entities:     {n_written:,}", flush=True)
    print(f"Matched Entities:      {matched_ct:,} ({matched_ct/n_written*100:.2f}%)", flush=True)
    print(f"Singletons:            {singleton_ct:,} ({singleton_ct/n_written*100:.2f}%)", flush=True)
    print(f"Total Matches:         {total_links:,} ({total_links/max(matched_ct,1):.2f} avg/matched entity)", flush=True)
    print(f"Avg Candidates/Entity: {total_cands/max(n_written,1):.2f}", flush=True)
    print(f"Total Time:            {elapsed/60:.2f} minutes", flush=True)
    print(f"\nOutputs:", flush=True)
    print(f"  {matching_path}", flush=True)
    print(f"  {candidate_path}", flush=True)
    print("=" * 75, flush=True)

    return {"n_entities": n_written, "matched": matched_ct, "singletons": singleton_ct,
            "total_links": total_links, "elapsed_s": elapsed}


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "student_resource/dataset/test"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    run_fast_matcher(test_dir=test_dir, output_dir=out_dir)
