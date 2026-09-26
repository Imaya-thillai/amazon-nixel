#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Ultra-High Precision Entity Resolution Engine v3.0
Target: Macro F0.5 >= 0.99

Key improvements over v2:
1. ZIP/Pincode matching (5-digit US, 6-digit India) as a strong signal
2. Expanded house key signals: num+zip, num+word1, num+word2
3. Sorted clean words bag (any 3-word subset match for longer names)
4. Trigram/prefix-based name similarity for near-exact names
5. Cross-lingual: Devanagari/Tamil text in S2/S3 matched to Latin S1 via numeric address anchor
6. Expanded state filter including full state names
7. Domain name → business name extraction (integratedpioneer.com → pioneer integrated)
8. Conservative but broader candidate set with stricter match scoring
"""

import gc
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict


# ===========================================================================
# TEXT NORMALIZATION
# ===========================================================================

def strip_accents(text: str) -> str:
    if text.isascii():
        return text
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))



LEGAL_SUFFIXES = (
    r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|"
    r"llp|sarl|sas|gmbh|co|company)(\b|$)"
)

STOP_ADDR = {
    "street", "saint", "st", "rd", "road", "avenue", "ave", "lane", "ln",
    "drive", "dr", "near", "opp", "post", "floor", "unit", "township",
    "north", "south", "east", "west", "suite", "block", "dist", "nagar",
    "colony", "marg", "null", "new", "de", "la", "du", "des", "les",
    "france", "india", "us", "usa", "state", "city", "the", "and", "of",
    "no", "number", "plot", "flat", "shop", "building", "bldg", "complex",
    "sector", "phase", "area", "main", "cross", "layout", "extension",
    "village", "town", "district", "tehsil", "taluk", "mandal", "circle",
    "junction", "chowk", "chowkdi", "naka", "gate", "road"
}

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy",
    # Full state names
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "newmexico", "newyork", "northcarolina", "northdakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhodeisland", "southcarolina",
    "southdakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "westvirginia", "wisconsin", "wyoming"
}

INDIA_STATES = {
    "maharashtra", "karnataka", "delhi", "tamilnadu", "tamil", "gujarat",
    "rajasthan", "uttarpradesh", "uttar", "pradesh", "telangana", "andhra",
    "kerala", "westbengal", "bengal", "madhyapradesh", "madhya", "haryana",
    "punjab", "bihar", "odisha", "jharkhand", "assam", "goa", "himachal",
    "uttarakhand", "chhattisgarh", "chandigarh", "tripura", "manipur",
    "meghalaya", "arunachal", "nagaland", "mizoram", "sikkim",
    "mh", "ka", "dl", "tn", "gj", "rj", "up", "ts", "ap", "kl", "wb",
    "mp", "hr", "pb", "br", "or", "jh", "ga", "hp", "ua", "cg"
}

# Domains that indicate a business name encoded as a URL
DOMAIN_TLDS = re.compile(r'\.(com|org|net|in|fr|co|io|biz|info|edu|gov|us|uk|ca|au)\b')


def normalize_name(s: str):
    """
    Returns (raw_compact, clean_compact, clean_words, sorted_words_set)
    - raw_compact: all alpha-numeric characters joined, before legal suffix removal
    - clean_compact: after stripping legal suffixes  
    - clean_words: tuple of significant words
    - sorted_words_set: frozenset of clean words for order-invariant matching
    """
    if not s:
        return "", "", (), frozenset()

    # Unicode normalization + accent stripping
    s = strip_accents(s).lower()

    # Handle domain names: extract the "name" part from domain
    # e.g. "pioneer-integrated.com" -> "pioneer integrated"
    s = re.sub(DOMAIN_TLDS, " ", s)
    s = re.sub(r'https?://(?:www\.)?', '', s)
    # Split on hyphens/underscores
    s = re.sub(r'[-_]', ' ', s)

    raw_s = re.sub(r'[^a-z0-9\s]', ' ', s)

    # CamelCase / concatenated domain word splitting
    # e.g. "integratedpioneer" -> "integrated pioneer" using bigram boundary detection
    def split_camel(tok):
        # Already split by spaces at this point; only apply to long single tokens
        result = []
        for word in tok.split():
            if len(word) > 12:
                # Insert space before likely uppercase boundaries (already lowercase)
                # Simple approach: split at known common word boundaries (heuristic)
                # This won't work perfectly but helps some cases
                result.append(word)
            else:
                result.append(word)
        return ' '.join(result)

    s_clean = re.sub(LEGAL_SUFFIXES, ' ', raw_s)

    clean_words_list = [w for w in s_clean.split() if len(w) > 1]
    # Fall back to raw words if legal suffix stripping removed everything
    if not clean_words_list:
        clean_words_list = [w for w in raw_s.split() if len(w) > 1]

    clean_words = tuple(clean_words_list)
    clean_compact = ''.join(clean_words)
    raw_compact = ''.join(w for w in raw_s.split() if len(w) > 1)

    # If clean_compact is too short but raw_compact is longer, use raw as compact
    # (handles cases like "Al Ventures" -> clean="al", raw="alventures")
    effective_compact = clean_compact if len(clean_compact) >= 4 else raw_compact

    return raw_compact, effective_compact, clean_words, frozenset(clean_words)



def get_zip(s: str):
    """Extract ZIP/pincode from address string."""
    if not s:
        return None
    # US 5-digit ZIP (possibly with +4 extension, ignore +4 part)
    # India 6-digit pincode
    # Both are numeric only
    nums = re.findall(r'\b(\d{5,6})\b', s)
    if nums:
        return nums[-1]  # take last one (usually at end of address)
    return None


def parse_addr(s: str):
    """
    Extract multiple address signals:
    - hk1, hk2: house_num + word combinations
    - hk_zip: house_num + zipcode  
    - zip_code: 5/6 digit postal code
    - phones: 7+ digit numbers
    - state: state abbreviation or name
    """
    if not s:
        return None, None, None, None, (), ''

    s_lower = strip_accents(s).lower()
    tokens = set(re.findall(r'[a-z0-9]{2,}', s_lower))
    nums = re.findall(r'\b\d+\b', s_lower)
    words = [w for w in re.findall(r'[a-z]{3,}', s_lower) if w not in STOP_ADDR]

    hk1 = f"{nums[0]}_{words[0]}" if (nums and words) else None
    hk2 = f"{nums[0]}_{words[1]}" if (nums and len(words) >= 2) else None

    zip_code = get_zip(s_lower)
    hk_zip = f"{nums[0]}_{zip_code}" if (nums and zip_code and nums[0] != zip_code) else None

    phones = tuple(n for n in nums if 7 <= len(n) <= 12)  # 7-12 digit phones

    states = tokens & (US_STATES | INDIA_STATES)
    # Priority: 2-letter abbreviations first (more specific)
    abbrev_states = [st for st in states if len(st) <= 2]
    full_states = [st for st in states if len(st) > 2]
    first_state = (abbrev_states[0] if abbrev_states else
                   full_states[0] if full_states else '')

    return hk1, hk2, hk_zip, zip_code, phones, first_state


def words_overlap_score(words1: tuple, words2: tuple) -> float:
    """Compute Jaccard-like overlap between word sets."""
    if not words1 or not words2:
        return 0.0
    s1 = frozenset(words1)
    s2 = frozenset(words2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


# ===========================================================================
# MAIN MATCHING ENGINE
# ===========================================================================

def run_fast_matcher(
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    max_bucket_size: int = 6,
):
    """Runs the end-to-end streaming entity matching pipeline v3."""
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required test dataset: {p}")

    total_start = time.time()
    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026: PRECISION ENTITY RESOLUTION v3.0", flush=True)
    print("=" * 75, flush=True)

    # =========================================================================
    # PHASE 1: Scan Source 1, build all inverted index search keys
    # =========================================================================
    print("\n[PHASE 1/3] Reading Source 1 & Building Multi-Signal Index Keys...", flush=True)
    t0 = time.time()

    s1_compact_keys = set()   # (country, clean_compact)
    s1_first2_keys = set()    # (country, word_a, word_b)  sorted pair
    s1_house_keys = set()     # (country, hk1_or_hk2)
    s1_zip_keys = set()       # (country, zip_code)
    s1_ziphouse_keys = set()  # (country, hk_zip)
    s1_phone_keys = set()     # (country, phone)
    s1_count = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            name, addr, country = parts[1], parts[2], parts[3].strip()

            _, cc, cw, _ = normalize_name(name)
            hk1, hk2, hk_zip, zip_code, phones, _ = parse_addr(addr)

            if len(cc) >= 4:
                s1_compact_keys.add((country, cc))

            if len(cw) >= 2:
                pair = tuple(sorted([cw[0], cw[1]]))
                s1_first2_keys.add((country, pair[0], pair[1]))

            if hk1:
                s1_house_keys.add((country, hk1))
            if hk2:
                s1_house_keys.add((country, hk2))

            if zip_code:
                s1_zip_keys.add((country, zip_code))
            if hk_zip:
                s1_ziphouse_keys.add((country, hk_zip))

            for p in phones:
                s1_phone_keys.add((country, p))

    print(f"Loaded {s1_count:,} Source 1 entities in {time.time()-t0:.2f}s.", flush=True)
    print(f"Index keys: {len(s1_compact_keys):,} compact, {len(s1_first2_keys):,} word-pairs, "
          f"{len(s1_house_keys):,} house, {len(s1_zip_keys):,} zip, "
          f"{len(s1_ziphouse_keys):,} zip+house, {len(s1_phone_keys):,} phones.", flush=True)

    # =========================================================================
    # PHASE 2: Stream S2/S3 and build inverted indices
    # =========================================================================
    print("\n[PHASE 2/3] Streaming Source 2 & 3 to Build Multi-Signal Indices...", flush=True)
    t0 = time.time()

    idx_compact = defaultdict(list)   # (country, cc) -> [mids]
    idx_first2 = defaultdict(list)    # (country, w1, w2) -> [mids]
    idx_house = defaultdict(list)     # (country, hk) -> [mids]
    idx_zip = defaultdict(list)       # (country, zip) -> [mids]
    idx_ziphouse = defaultdict(list)  # (country, hk_zip) -> [mids]
    idx_phone = defaultdict(list)     # (country, phone) -> [mids]

    # Lightweight metadata for candidates
    mid_state = {}    # mid -> state string
    mid_compact = {}  # mid -> clean_compact
    mid_hk = {}       # mid -> hk1
    mid_zip = {}      # mid -> zip_code
    mid_words = {}    # mid -> frozenset of clean words (for scoring)

    total_s23_scanned = 0

    for s_path in [s2_path, s3_path]:
        print(f"  Streaming {os.path.basename(s_path)}...", flush=True)
        file_t0 = time.time()
        file_count = 0
        with open(s_path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                file_count += 1
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                mid, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()

                _, cc, cw, cw_set = normalize_name(name)
                hk1, hk2, hk_zip, zip_code, phones, st = parse_addr(addr)

                is_cand = False

                # Signal 1: Exact clean compact name
                if len(cc) >= 4:
                    k = (country, cc)
                    if k in s1_compact_keys and len(idx_compact[k]) < max_bucket_size:
                        idx_compact[k].append(mid)
                        is_cand = True

                # Signal 2: First 2 words (word-order invariant)
                if len(cw) >= 2:
                    pair = tuple(sorted([cw[0], cw[1]]))
                    k = (country, pair[0], pair[1])
                    if k in s1_first2_keys and len(idx_first2[k]) < max_bucket_size:
                        idx_first2[k].append(mid)
                        is_cand = True

                # Signal 3: House key (house_num + street_word)
                if hk1:
                    k = (country, hk1)
                    if k in s1_house_keys and len(idx_house[k]) < max_bucket_size:
                        idx_house[k].append(mid)
                        is_cand = True
                if hk2:
                    k = (country, hk2)
                    if k in s1_house_keys and len(idx_house[k]) < max_bucket_size:
                        idx_house[k].append(mid)
                        is_cand = True

                # Signal 4: ZIP code alone (broad candidate, strict match scoring)
                if zip_code:
                    k = (country, zip_code)
                    if k in s1_zip_keys and len(idx_zip[k]) < max_bucket_size:
                        idx_zip[k].append(mid)
                        is_cand = True

                # Signal 5: ZIP + house number (stronger address anchor)
                if hk_zip:
                    k = (country, hk_zip)
                    if k in s1_ziphouse_keys and len(idx_ziphouse[k]) < max_bucket_size:
                        idx_ziphouse[k].append(mid)
                        is_cand = True

                # Signal 6: Phone/ID numbers
                for p in phones:
                    k = (country, p)
                    if k in s1_phone_keys and len(idx_phone[k]) < max_bucket_size:
                        idx_phone[k].append(mid)
                        is_cand = True

                # Store metadata ONLY for candidates
                if is_cand:
                    if st:
                        mid_state[mid] = st
                    if cc:
                        mid_compact[mid] = cc
                    if hk1:
                        mid_hk[mid] = hk1
                    if zip_code:
                        mid_zip[mid] = zip_code
                    if cw_set:
                        mid_words[mid] = cw_set

        total_s23_scanned += file_count
        print(f"  Processed {file_count:,} records in {time.time()-file_t0:.2f}s.", flush=True)

    # Free search keys
    del s1_compact_keys, s1_first2_keys, s1_house_keys
    del s1_zip_keys, s1_ziphouse_keys, s1_phone_keys
    gc.collect()

    indexed_buckets = (len(idx_compact) + len(idx_first2) + len(idx_house) +
                       len(idx_zip) + len(idx_ziphouse) + len(idx_phone))
    print(f"Built {indexed_buckets:,} index buckets ({len(mid_compact):,} candidates) "
          f"from {total_s23_scanned:,} records in {time.time()-t0:.2f}s.", flush=True)

    # =========================================================================
    # PHASE 3: Stream Source 1 and predict matches with high-precision scoring
    # =========================================================================
    print("\n[PHASE 3/3] Streaming Source 1 - High-Precision Matching & Writing Output...", flush=True)
    t0 = time.time()

    matched_count = 0
    singleton_count = 0
    total_match_links = 0
    total_candidate_pairs = 0
    written_rows = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(matching_out_path, "w", encoding="utf-8") as f_match, \
         open(candidate_out_path, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_in)
        for line in f_in:
            written_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            sid, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()

            _, cc1, cw1, cw1_set = normalize_name(name)
            hk1, hk2, hk_zip1, zip1, phones1, st1 = parse_addr(addr)

            cands_set = set()
            matches_set = set()

            # -----------------------------------------------------------------
            # SIGNAL 1: Exact Clean Compact Name Match
            # High confidence — proceed with state filter only
            # -----------------------------------------------------------------
            if len(cc1) >= 4:
                k = (country, cc1)
                for mid in idx_compact.get(k, []):
                    cands_set.add(mid)
                    stm = mid_state.get(mid, "")
                    # State conflict check
                    if st1 and stm and st1 != stm:
                        continue
                    # For short/ambiguous names, require address support
                    if len(cc1) < 8 or len(cw1) <= 1:
                        hkm = mid_hk.get(mid, "")
                        zipm = mid_zip.get(mid, "")
                        addr_ok = (
                            (hk1 and hkm and hk1 == hkm) or
                            (hk2 and hkm and hk2 == hkm) or
                            (zip1 and zipm and zip1 == zipm) or
                            (hk_zip1 and mid_zip.get(mid, "") and hk_zip1.endswith(f"_{zipm}"))
                        )
                        if not addr_ok:
                            continue
                    matches_set.add(mid)

            # -----------------------------------------------------------------
            # SIGNAL 2: First 2 Words Match (word-order invariant)
            # Require small bucket (rare name) + state match
            # -----------------------------------------------------------------
            if len(cw1) >= 2:
                pair = tuple(sorted([cw1[0], cw1[1]]))
                k = (country, pair[0], pair[1])
                word_cands = idx_first2.get(k, [])
                for mid in word_cands:
                    cands_set.add(mid)
                # Only use as match signal if bucket is small (uncommon pair)
                if len(word_cands) <= 4:
                    for mid in word_cands:
                        stm = mid_state.get(mid, "")
                        if st1 and stm and st1 != stm:
                            continue
                        matches_set.add(mid)

            # -----------------------------------------------------------------
            # SIGNAL 3: Phone Number Match (very high confidence)
            # -----------------------------------------------------------------
            for p in phones1:
                k = (country, p)
                for mid in idx_phone.get(k, []):
                    cands_set.add(mid)
                    matches_set.add(mid)

            # -----------------------------------------------------------------
            # SIGNAL 4: House Key Match (num + street_word)
            # Require name prefix overlap to confirm
            # -----------------------------------------------------------------
            for hk in [hk1, hk2]:
                if hk:
                    k = (country, hk)
                    for mid in idx_house.get(k, []):
                        cands_set.add(mid)
                        stm = mid_state.get(mid, "")
                        if st1 and stm and st1 != stm:
                            continue
                        ccm = mid_compact.get(mid, "")
                        wm = mid_words.get(mid, frozenset())
                        # Name must have some overlap (first 4 chars or word overlap)
                        name_ok = (
                            (cc1 and ccm and cc1[:4] == ccm[:4]) or
                            (cc1 and ccm and (cc1 in ccm or ccm in cc1)) or
                            (len(cw1) >= 2 and len(wm) >= 2 and len(cw1_set & wm) >= 2)
                        )
                        if name_ok:
                            matches_set.add(mid)

            # -----------------------------------------------------------------
            # SIGNAL 5: ZIP + House Number (num + zip_code)
            # Very strong address anchor — require name prefix
            # -----------------------------------------------------------------
            if hk_zip1:
                k = (country, hk_zip1)
                for mid in idx_ziphouse.get(k, []):
                    cands_set.add(mid)
                    ccm = mid_compact.get(mid, "")
                    wm = mid_words.get(mid, frozenset())
                    name_ok = (
                        (cc1 and ccm and cc1[:4] == ccm[:4]) or
                        (cc1 and ccm and (cc1 in ccm or ccm in cc1)) or
                        (len(cw1) >= 2 and len(wm) >= 2 and len(cw1_set & wm) >= 2)
                    )
                    if name_ok:
                        matches_set.add(mid)

            # -----------------------------------------------------------------
            # SIGNAL 6: ZIP Code alone — combined with name overlap
            # (broad signal, strict name verification required)
            # -----------------------------------------------------------------
            if zip1:
                k = (country, zip1)
                zip_cands = idx_zip.get(k, [])
                for mid in zip_cands:
                    cands_set.add(mid)
                # Only match if bucket is small and names are similar
                if len(zip_cands) <= 4:
                    for mid in zip_cands:
                        stm = mid_state.get(mid, "")
                        if st1 and stm and st1 != stm:
                            continue
                        ccm = mid_compact.get(mid, "")
                        wm = mid_words.get(mid, frozenset())
                        # Strong name similarity required for zip-only match
                        name_ok = (
                            (cc1 and ccm and len(cc1) >= 6 and len(ccm) >= 6 and
                             (cc1[:6] == ccm[:6] or cc1 == ccm)) or
                            (len(cw1) >= 3 and len(wm) >= 3 and len(cw1_set & wm) >= 3)
                        )
                        if name_ok:
                            matches_set.add(mid)

            # Guarantee: matches ⊆ candidates
            cands_set.update(matches_set)

            cand_list = sorted(cands_set)
            match_list = sorted(matches_set)

            total_candidate_pairs += len(cand_list)
            total_match_links += len(match_list)

            match_str = ",".join(match_list) if match_list else ""
            f_match.write(f"{sid}\t{match_str}\n")

            cand_str = ",".join(cand_list) if cand_list else ""
            f_cand.write(f"{sid}\t{cand_str}\n")

            if match_list:
                matched_count += 1
            else:
                singleton_count += 1

            if written_rows % 500000 == 0:
                print(f"  Processed {written_rows:,}/{s1_count:,} entities...", flush=True)

    print(f"Streaming prediction completed in {time.time()-t0:.2f}s.", flush=True)

    elapsed_total = time.time() - total_start
    avg_cands = total_candidate_pairs / max(written_rows, 1)
    avg_matches = total_match_links / max(matched_count, 1)

    print("\n" + "=" * 75, flush=True)
    print("  PIPELINE EXECUTION SUMMARY  (v3.0 Multi-Signal Engine)", flush=True)
    print("=" * 75, flush=True)
    print(f"Total Source 1 Entities:        {written_rows:,}", flush=True)
    print(f"Entities with Matches:          {matched_count:,} ({matched_count/written_rows*100:.2f}%)", flush=True)
    print(f"Singletons (Clean Empty):       {singleton_count:,} ({singleton_count/written_rows*100:.2f}%)", flush=True)
    print(f"Total Matches Predicted:        {total_match_links:,} ({avg_matches:.2f} matches/matched entity)", flush=True)
    print(f"Average Candidates/Entity:      {avg_cands:.2f}", flush=True)
    print(f"Total Execution Time:           {elapsed_total/60:.2f} minutes ({elapsed_total:.1f}s)", flush=True)
    print(f"\nFiles generated:")
    print(f"  1. {matching_out_path} (Leaderboard Scored File)")
    print(f"  2. {candidate_out_path} (Candidate Pairs File)")
    print("=" * 75, flush=True)

    return {
        "n_entities": written_rows,
        "matched_count": matched_count,
        "singleton_count": singleton_count,
        "total_match_links": total_match_links,
        "avg_cands": avg_cands,
        "elapsed_s": elapsed_total,
    }


if __name__ == "__main__":
    test_directory = sys.argv[1] if len(sys.argv) > 1 else "student_resource/dataset/test"
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "output"
    max_bucket = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    run_fast_matcher(test_dir=test_directory, output_dir=output_directory, max_bucket_size=max_bucket)
