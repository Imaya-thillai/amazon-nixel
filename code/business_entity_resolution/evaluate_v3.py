#!/usr/bin/env python3
"""
Quick benchmark: run new v3 engine on training data and compute Macro F0.5.
Usage: python evaluate_v3.py [n_eval=20000]
"""

import csv
import re
import sys
import unicodedata
from collections import defaultdict

# Import v3 engine functions
sys.path.insert(0, 'code/business_entity_resolution/src')
from fast_matcher import normalize_name, parse_addr


def f_beta(precision, recall, beta=0.5):
    b2 = beta ** 2
    if precision + recall == 0:
        return 0.0
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def main():
    train_dir = 'student_resource/dataset/train'
    max_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    max_bucket = 6  # v3 default

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
    s1_zip_keys = set()
    s1_ziphouse_keys = set()
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
            _, cc, cw, cw_set = normalize_name(name)
            hk1, hk2, hk_zip, zip_code, phones, st = parse_addr(addr)
            s1_data[sid] = (name, addr, country, cc, cw, cw_set, hk1, hk2, hk_zip, zip_code, phones, st)

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

    print(f'S1 loaded: {len(s1_data)} entities')

    idx_compact = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_house = defaultdict(list)
    idx_zip = defaultdict(list)
    idx_ziphouse = defaultdict(list)
    idx_phone = defaultdict(list)
    mid_state = {}
    mid_compact = {}
    mid_hk = {}
    mid_zip = {}
    mid_words = {}

    for src_file in ['train_source2.tsv', 'train_source3.tsv']:
        print(f'  Indexing {src_file}...')
        with open(f'{train_dir}/{src_file}', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                mid, name, addr, country = row[0], row[1], row[2], row[3].strip()
                _, cc, cw, cw_set = normalize_name(name)
                hk1, hk2, hk_zip, zip_code, phones, st = parse_addr(addr)
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
                if zip_code:
                    k = (country, zip_code)
                    if k in s1_zip_keys and len(idx_zip[k]) < max_bucket:
                        idx_zip[k].append(mid)
                        is_cand = True
                if hk_zip:
                    k = (country, hk_zip)
                    if k in s1_ziphouse_keys and len(idx_ziphouse[k]) < max_bucket:
                        idx_ziphouse[k].append(mid)
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
                    if zip_code:
                        mid_zip[mid] = zip_code
                    if cw_set:
                        mid_words[mid] = cw_set

    print(f'Evaluating {max_eval} entities (v3 engine)...')

    f05_scores = []
    recall_zero_ct = 0
    precision_zero_ct = 0
    fn_ct = 0
    fp_ct = 0
    eval_count = 0
    tp_total = 0

    for sid, (name, addr, country, cc1, cw1, cw1_set, hk1, hk2, hk_zip1, zip1, phones1, st1) in list(s1_data.items())[:max_eval]:
        gt_set = gt.get(sid, set()) - {''}
        eval_count += 1

        cands_set = set()
        matches_set = set()

        # Signal 1: Exact compact name
        if len(cc1) >= 4:
            k = (country, cc1)
            for mid in idx_compact.get(k, []):
                cands_set.add(mid)
                stm = mid_state.get(mid, '')
                if st1 and stm and st1 != stm:
                    continue
                if len(cc1) < 8 or len(cw1) <= 1:
                    hkm = mid_hk.get(mid, '')
                    zipm = mid_zip.get(mid, '')
                    addr_ok = (
                        (hk1 and hkm and hk1 == hkm) or
                        (hk2 and hkm and hk2 == hkm) or
                        (zip1 and zipm and zip1 == zipm)
                    )
                    if not addr_ok:
                        continue
                matches_set.add(mid)

        # Signal 2: First 2 words
        if len(cw1) >= 2:
            pair = tuple(sorted([cw1[0], cw1[1]]))
            k = (country, pair[0], pair[1])
            word_cands = idx_first2.get(k, [])
            for mid in word_cands:
                cands_set.add(mid)
            if len(word_cands) <= 4:
                for mid in word_cands:
                    stm = mid_state.get(mid, '')
                    if st1 and stm and st1 != stm:
                        continue
                    matches_set.add(mid)

        # Signal 3: Phone
        for p in phones1:
            k = (country, p)
            for mid in idx_phone.get(k, []):
                cands_set.add(mid)
                matches_set.add(mid)

        # Signal 4: House key
        for hk in [hk1, hk2]:
            if hk:
                k = (country, hk)
                for mid in idx_house.get(k, []):
                    cands_set.add(mid)
                    stm = mid_state.get(mid, '')
                    if st1 and stm and st1 != stm:
                        continue
                    ccm = mid_compact.get(mid, '')
                    wm = mid_words.get(mid, frozenset())
                    name_ok = (
                        (cc1 and ccm and cc1[:4] == ccm[:4]) or
                        (cc1 and ccm and (cc1 in ccm or ccm in cc1)) or
                        (len(cw1) >= 2 and len(wm) >= 2 and len(cw1_set & wm) >= 2)
                    )
                    if name_ok:
                        matches_set.add(mid)

        # Signal 5: ZIP + house
        if hk_zip1:
            k = (country, hk_zip1)
            for mid in idx_ziphouse.get(k, []):
                cands_set.add(mid)
                ccm = mid_compact.get(mid, '')
                wm = mid_words.get(mid, frozenset())
                name_ok = (
                    (cc1 and ccm and cc1[:4] == ccm[:4]) or
                    (cc1 and ccm and (cc1 in ccm or ccm in cc1)) or
                    (len(cw1) >= 2 and len(wm) >= 2 and len(cw1_set & wm) >= 2)
                )
                if name_ok:
                    matches_set.add(mid)

        # Signal 6: ZIP alone
        if zip1:
            k = (country, zip1)
            zip_cands = idx_zip.get(k, [])
            for mid in zip_cands:
                cands_set.add(mid)
            if len(zip_cands) <= 4:
                for mid in zip_cands:
                    stm = mid_state.get(mid, '')
                    if st1 and stm and st1 != stm:
                        continue
                    ccm = mid_compact.get(mid, '')
                    wm = mid_words.get(mid, frozenset())
                    name_ok = (
                        (cc1 and ccm and len(cc1) >= 6 and len(ccm) >= 6 and
                         (cc1[:6] == ccm[:6] or cc1 == ccm)) or
                        (len(cw1) >= 3 and len(wm) >= 3 and len(cw1_set & wm) >= 3)
                    )
                    if name_ok:
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
            tp_total += tp
            if rec == 0:
                recall_zero_ct += 1
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
    print(f'Engine: V3 Multi-Signal')
    print(f'Evaluated: {eval_count:,} entities')
    print(f'Macro F0.5: {macro_f05:.6f}')
    print(f'Total TP: {tp_total:,}')
    print(f'Total FP (wrong matches): {fp_ct:,}')
    print(f'Total FN (missed matches): {fn_ct:,}')
    print(f'Entities with recall=0 (missed entirely): {recall_zero_ct:,} ({100*recall_zero_ct/max(eval_count,1):.2f}%)')
    print(f'Entities with precision=0 (all FP): {precision_zero_ct:,} ({100*precision_zero_ct/max(eval_count,1):.2f}%)')
    print(f'========================================')


if __name__ == '__main__':
    main()
