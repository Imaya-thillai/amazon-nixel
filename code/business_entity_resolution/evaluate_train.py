#!/usr/bin/env python3
"""Evaluate current matching engine on training data and compute true Macro F0.5."""

import csv
import re
import sys
import unicodedata
from collections import defaultdict

def strip_accents(text):
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))

LEGAL_SUFFIXES = (
    r'\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|'
    r'llp|sarl|sas|sa|gmbh|co|company|enterprises|enterprise|group|'
    r'services|associates|solutions|technologies|tech|holdings)\b'
)

STOP_ADDR = {
    'street', 'saint', 'st', 'rd', 'road', 'avenue', 'ave', 'lane', 'ln',
    'drive', 'dr', 'near', 'opp', 'post', 'floor', 'unit', 'township',
    'north', 'south', 'east', 'west', 'suite', 'block', 'dist', 'nagar',
    'colony', 'marg', 'null', 'new', 'de', 'la', 'du', 'des', 'les',
    'france', 'india', 'us', 'usa', 'state', 'city'
}

US_STATES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id',
    'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms',
    'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok',
    'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv',
    'wi', 'wy'
}

INDIA_STATES = {
    'maharashtra', 'karnataka', 'delhi', 'tamil', 'nadu', 'gujarat', 'rajasthan',
    'uttar', 'pradesh', 'telangana', 'andhra', 'kerala', 'west', 'bengal',
    'madhya', 'haryana', 'punjab', 'bihar', 'odisha', 'jharkhand', 'assam',
    'mh', 'ka', 'dl', 'tn', 'gj', 'rj', 'up', 'ts', 'ap', 'kl', 'wb', 'mp', 'hr', 'pb'
}


def normalize_name(s):
    if not s:
        return '', '', ()
    s = strip_accents(s).lower()
    s = re.sub(r'https?://(?:www\.)?', '', s)
    s = re.sub(r'\.(com|org|net|in|fr|co|io|biz|info|edu|gov)\b', '', s)
    raw_s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s_clean = re.sub(LEGAL_SUFFIXES, '', raw_s)
    clean_words = tuple(w for w in s_clean.split() if len(w) > 1)
    clean_compact = ''.join(clean_words)
    return ''.join(w for w in raw_s.split() if len(w) > 1), clean_compact, clean_words


def parse_addr(s):
    if not s:
        return None, None, (), ''
    s = strip_accents(s).lower()
    tokens = set(re.findall(r'[a-z0-9]{2,}', s))
    nums = re.findall(r'\b\d+\b', s)
    words = [w for w in re.findall(r'[a-z]{3,}', s) if w not in STOP_ADDR]
    hk1 = f'{nums[0]}_{words[0]}' if (nums and words) else None
    hk2 = f'{nums[0]}_{words[1]}' if (nums and len(words) >= 2) else None
    phones = tuple(n for n in nums if len(n) >= 7)
    states = tokens & (US_STATES | INDIA_STATES)
    first_state = list(states)[0] if states else ''
    return hk1, hk2, phones, first_state


def f_beta(precision, recall, beta=0.5):
    b2 = beta ** 2
    if precision + recall == 0:
        return 0.0
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def main():
    train_dir = 'student_resource/dataset/train'
    max_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    max_bucket = 4

    print(f'Loading ground truth...')
    gt = {}
    with open(f'{train_dir}/train_ground_truth.tsv', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                gt[row[0]] = set(row[1].strip().split(',')) if row[1].strip() else set()
    print(f'GT: {len(gt)} entities')

    print('Building S1 keys...')
    s1_compact_keys = set()
    s1_first2_keys = set()
    s1_house_keys = set()
    s1_phone_keys = set()
    s1_data = {}

    with open(f'{train_dir}/train_source1.tsv', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            sid, name, addr, country = row[0], row[1], row[2], row[3].strip()
            if sid not in gt:
                continue
            _, cc, cw = normalize_name(name)
            hk1, hk2, phones, st = parse_addr(addr)
            s1_data[sid] = (name, addr, country, cc, cw, hk1, hk2, phones, st)
            if len(cc) >= 4:
                s1_compact_keys.add((country, cc))
            if len(cw) >= 2:
                pair = tuple(sorted([cw[0], cw[1]]))
                s1_first2_keys.add((country, pair[0], pair[1]))
            if hk1:
                s1_house_keys.add((country, hk1))
            if hk2:
                s1_house_keys.add((country, hk2))
            for p in phones:
                s1_phone_keys.add((country, p))

    print(f'S1 loaded: {len(s1_data)} entities')

    idx_compact = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_house = defaultdict(list)
    idx_phone = defaultdict(list)
    mid_state = {}
    mid_compact = {}
    mid_hk = {}

    for src_file in ['train_source2.tsv', 'train_source3.tsv']:
        print(f'  Indexing {src_file}...')
        with open(f'{train_dir}/{src_file}', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                mid, name, addr, country = row[0], row[1], row[2], row[3].strip()
                _, cc, cw = normalize_name(name)
                hk1, hk2, phones, st = parse_addr(addr)
                is_cand = False
                if len(cc) >= 4:
                    k = (country, cc)
                    if k in s1_compact_keys and len(idx_compact[k]) < max_bucket:
                        idx_compact[k].append(mid)
                        is_cand = True
                if len(cw) >= 2:
                    pair = tuple(sorted([cw[0], cw[1]]))
                    k = (country, pair[0], pair[1])
                    if k in s1_first2_keys and len(idx_first2[k]) < max_bucket:
                        idx_first2[k].append(mid)
                        is_cand = True
                if hk1:
                    k = (country, hk1)
                    if k in s1_house_keys and len(idx_house[k]) < max_bucket:
                        idx_house[k].append(mid)
                        is_cand = True
                if hk2:
                    k = (country, hk2)
                    if k in s1_house_keys and len(idx_house[k]) < max_bucket:
                        idx_house[k].append(mid)
                        is_cand = True
                for p in phones:
                    k = (country, p)
                    if k in s1_phone_keys and len(idx_phone[k]) < max_bucket:
                        idx_phone[k].append(mid)
                        is_cand = True
                if is_cand:
                    if st:
                        mid_state[mid] = st
                    if cc:
                        mid_compact[mid] = cc
                    if hk1:
                        mid_hk[mid] = hk1

    print(f'Index built. Evaluating {max_eval} entities...')

    f05_scores = []
    recall_zero_ct = 0
    precision_zero_ct = 0
    fn_ct = 0
    fp_ct = 0
    eval_count = 0
    missed_examples = []

    for sid, (name, addr, country, cc1, cw1, hk1, hk2, phones1, st1) in list(s1_data.items())[:max_eval]:
        gt_set = gt.get(sid, set()) - {''}
        eval_count += 1

        cands_set = set()
        matches_set = set()

        if len(cc1) >= 4:
            k = (country, cc1)
            for mid in idx_compact.get(k, []):
                cands_set.add(mid)
                stm = mid_state.get(mid, '')
                if st1 and stm and st1 != stm:
                    continue
                if len(cw1) <= 1 or len(cc1) < 8:
                    hkm = mid_hk.get(mid, '')
                    if hk1 and hkm and hk1 != hkm and (hk2 != hkm if hk2 else True):
                        continue
                matches_set.add(mid)

        if len(cw1) >= 2:
            pair = tuple(sorted([cw1[0], cw1[1]]))
            k = (country, pair[0], pair[1])
            word_cands = idx_first2.get(k, [])
            for mid in word_cands:
                cands_set.add(mid)
            if len(word_cands) <= 3:
                for mid in word_cands:
                    stm = mid_state.get(mid, '')
                    if st1 and stm and st1 != stm:
                        continue
                    matches_set.add(mid)

        for p in phones1:
            k = (country, p)
            for mid in idx_phone.get(k, []):
                cands_set.add(mid)
                matches_set.add(mid)

        for hk in [hk1, hk2]:
            if hk:
                k = (country, hk)
                for mid in idx_house.get(k, []):
                    cands_set.add(mid)
                    stm = mid_state.get(mid, '')
                    if st1 and stm and st1 != stm:
                        continue
                    ccm = mid_compact.get(mid, '')
                    if not cc1 or not ccm or cc1[:3] == ccm[:3] or (cc1 in ccm) or (ccm in cc1):
                        matches_set.add(mid)

        pred_set = matches_set

        if gt_set:
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = f_beta(prec, rec)
            f05_scores.append(score)
            fn_ct += fn
            fp_ct += fp
            if rec == 0:
                recall_zero_ct += 1
                if len(missed_examples) < 5:
                    missed_examples.append((sid, name, addr, gt_set, pred_set))
            if prec == 0:
                precision_zero_ct += 1
        else:
            if not pred_set:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                precision_zero_ct += 1

    macro_f05 = sum(f05_scores) / len(f05_scores)
    print(f'\n========================================')
    print(f'Evaluated {eval_count} entities')
    print(f'Macro F0.5: {macro_f05:.6f}')
    print(f'Entities with recall=0 (missed entirely): {recall_zero_ct} ({100*recall_zero_ct/max(eval_count,1):.2f}%)')
    print(f'Entities with precision=0 (all FP): {precision_zero_ct} ({100*precision_zero_ct/max(eval_count,1):.2f}%)')
    print(f'Total FN (missed true matches): {fn_ct}')
    print(f'Total FP (wrong matches): {fp_ct}')
    print(f'========================================')

    if missed_examples:
        print('\nSample entities with recall=0 (completely missed):')
        for sid, name, addr, gt_set, pred_set in missed_examples:
            print(f'  S1: {sid} | {name[:50]} | {addr[:50]}')
            print(f'    GT: {list(gt_set)[:3]}')
            print(f'    Predicted: {list(pred_set)[:3]}')
            print()


if __name__ == '__main__':
    main()
