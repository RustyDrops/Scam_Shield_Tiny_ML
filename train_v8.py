#!/usr/bin/env python3
"""Scam Shield v8: midpoint threshold cap + task-scam-space families.

Pre-registered in PROVENANCE.md (v8 section) BEFORE any v8 training.

v7's false-alarm fix worked perfectly (test FPR 0.0029, transactional
FP 0.0000) but its strict 0.30% validation cap cost recall everywhere
(modern 0.9619 vs the 0.965 gate; obfuscation edge gone on fresh set
B-v7), and its exam exposed a transfer gap: never-seen task_job
scored 0.5714 — task/gig/crypto-unlock language does not transfer
from the other families.

v8 = v7's recipe (incl. the hard-legit group and its
template-disjoint validation slice, unchanged) with three changes:
1. Threshold chosen on VALIDATION ONLY, maximising scam recall
   subject to validation legit FPR <= 0.0040 (midpoint between v6's
   loose 1.0% and v7's strict 0.30%). Fallback: lowest-val-FPR
   threshold, recorded.
2. NEW training families gig_task + crypto_unlock (seeds_v8.json,
   government-publication mechanics, v4 slot-template method) join
   the family groups and recombinations. toll/task_job return to
   normal training (spent exams).
3. FRESH held-out exam, built before training (gen_v8_data.py),
   each set evaluated ONCE:
   - obfuscation set C: bases disjoint from v6's eval AND B-v7
     (765 IMC25 test positives / 78 UCI test ham remain), set-B
     mechanisms at B-v7 variants, fresh seeds;
   - held-out families romance + grandparent (NEW in seeds_v8.json;
     excluded from ALL training/validation incl. recombinations);
   - v5 scored once on each as baseline; frozen v5 suites once as
     regression.

Keeper gates (pre-registered, ALL must pass): modern recall >=
0.965; legit FPR <= 0.005; transactional legit FP <= 0.01; obf-C
recall STRICTLY > v5's on set C; held-out-family (romance +
grandparent) recall >= v5's on the same families.
"""
import json
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import augment_v4
import obfuscate_v6
import predict_v5
from train_v2 import load_uci, load_modern, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
OBF_SCAM_COPIES = 4500
OBF_LEGIT_COPIES = 900
VAL_FPR_CAP = 0.0040


def load_jsonl(*parts):
    path = os.path.join(HERE, *parts)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def recall_block(probs, thr):
    return round(float((np.asarray(probs) >= thr).mean()), 4)


def main():
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED,
                              stratify=y[tmp])
    X = lambda ii: [texts[i] for i in ii]
    uci_val_test_keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, mstats = load_modern(uci_val_test_keys)

    # v8: NO v4 family is held out (toll/task_job were v7's spent exam)
    fam_train = load_jsonl("data", "v4", "fam_train.jsonl")
    fam_val = load_jsonl("data", "v4", "fam_validation.jsonl")
    fam_test = load_jsonl("data", "v4", "fam_test.jsonl")
    fam8_train = load_jsonl("data", "v8", "fam8_train.jsonl")
    fam8_val = load_jsonl("data", "v8", "fam8_validation.jsonl")
    fam8_test = load_jsonl("data", "v8", "fam8_test.jsonl")
    fam_test_all = fam_test + fam8_test
    demasked_train = [r["text"] for r in
                      load_jsonl("data", "v4", "imc25_train_demasked.jsonl")]
    txn_train = load_jsonl("data", "v5", "txn_train.jsonl")
    txn_val = load_jsonl("data", "v5", "txn_validation.jsonl")
    txn_test = load_jsonl("data", "v5", "txn_test.jsonl")
    txn_train_legit = [r for r in txn_train if r["label"] == 0]
    txn_train_scam = [r for r in txn_train if r["label"] == 1]
    recomb_train = load_jsonl("data", "v8", "recomb_train.jsonl")
    recomb_val = load_jsonl("data", "v8", "recomb_validation.jsonl")
    obf_scam_eval = load_jsonl("data", "v8", "obf_eval_scam.jsonl")
    obf_legit_eval = load_jsonl("data", "v8", "obf_eval_legit.jsonl")
    hf_eval = load_jsonl("data", "v8", "hf_eval.jsonl")
    legit7_train = load_jsonl("data", "v7", "legit_train.jsonl")
    legit7_val = load_jsonl("data", "v7", "legit_validation.jsonl")

    uci_train_ham = [texts[i] for i in tr if y[i] == 0]
    uci_train_spam = [texts[i] for i in tr if y[i] == 1]

    # ---- obfuscated copies (set A transforms, seeded; v6/v7 recipe) ----
    rng = random.Random("v8-obf-train")
    scam_pool = (modern["train"] + demasked_train
                 + [r["text"] for r in fam_train]
                 + [r["text"] for r in fam8_train]
                 + [r["text"] for r in recomb_train]
                 + [r["text"] for r in txn_train_scam])
    rng.shuffle(scam_pool)
    obf_scam = [obfuscate_v6.obfuscate_train_scam(t, rng)
                for t in scam_pool[:OBF_SCAM_COPIES]]
    legit_pool = (uci_train_ham + [r["text"] for r in txn_train_legit]
                  + [r["text"] for r in legit7_train])
    rng.shuffle(legit_pool)
    obf_legit = [obfuscate_v6.obfuscate_train_legit(t, rng)
                 for t in legit_pool[:OBF_LEGIT_COPIES]]

    group_texts = [
        uci_train_ham,                                   # 0 legit
        uci_train_spam,                                  # 1
        modern["train"],                                 # 2
        demasked_train,                                  # 3
        [r["text"] for r in fam_train],                  # 4 (v4 families)
        [r["text"] for r in fam8_train],                 # 5 (NEW families)
        [r["text"] for r in txn_train_legit],            # 6 legit
        [r["text"] for r in txn_train_scam],             # 7
        [r["text"] for r in recomb_train],               # 8
        obf_scam,                                        # 9
        obf_legit,                                       # 10 legit
        [r["text"] for r in legit7_train],               # 11 legit (v7)
    ]
    group_labels = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    train_texts, train_y_list, groups = [], [], []
    for g, (gt, lab) in enumerate(zip(group_texts, group_labels)):
        train_texts += gt
        train_y_list += [lab] * len(gt)
        groups += [g] * len(gt)
    train_y = np.array(train_y_list)
    groups = np.array(groups)
    sample_w = np.zeros(len(groups))
    for g in range(len(group_texts)):
        m = groups == g
        sample_w[m] = len(groups) / (float(len(group_texts)) * m.sum())

    val_texts = (X(va) + modern["validation"]
                 + [r["text"] for r in fam_val]
                 + [r["text"] for r in fam8_val]
                 + [r["text"] for r in txn_val]
                 + [r["text"] for r in recomb_val]
                 + [r["text"] for r in legit7_val])
    val_y = np.concatenate([y[va],
                            np.ones(len(modern["validation"]), dtype=int),
                            np.ones(len(fam_val), dtype=int),
                            np.ones(len(fam8_val), dtype=int),
                            np.array([r["label"] for r in txn_val]),
                            np.ones(len(recomb_val), dtype=int),
                            np.zeros(len(legit7_val), dtype=int)])
    val_legit_n = int((val_y == 0).sum())

    # ---- leakage asserts (exam material must not be in train/val) ----
    train_keys = {t.strip().lower() for t in train_texts}
    val_keys = {t.strip().lower() for t in val_texts}
    for name, rows in (("hf_eval", hf_eval),
                       ("obf_eval_scam", obf_scam_eval),
                       ("obf_eval_legit", obf_legit_eval)):
        for r in rows:
            k = r["text"].strip().lower()
            assert k not in train_keys, f"{name} row leaked into train"
            assert k not in val_keys, f"{name} row leaked into val"
    assert not {r["family"] for r in fam_train + fam_val} & \
        {"romance", "grandparent"}
    print("LEAKAGE_ASSERTS_PASSED", flush=True)

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                           max_features=15000, min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w[\w']+\b")
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb",
                           ngram_range=(3, 6), max_features=35000, min_df=2,
                           sublinear_tf=True)
    vec = FeatureUnion([("word", word), ("char", char)])
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000,
                             solver="liblinear")
    clf.fit(Xtr, train_y, sample_weight=sample_w)

    # ---- threshold: max recall s.t. validation legit FPR <= cap ----
    pv = clf.predict_proba(vec.transform(val_texts))[:, 1]
    best = None
    lowest_fpr = None
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics(val_y, pv, thr)
        if lowest_fpr is None or \
                m["legit_false_positive_rate"] < \
                lowest_fpr["legit_false_positive_rate"]:
            lowest_fpr = m
        if m["legit_false_positive_rate"] <= VAL_FPR_CAP:
            if best is None or m["scam_recall"] > best["scam_recall"]:
                best = m
    cap_satisfied = best is not None
    if not cap_satisfied:
        best = lowest_fpr
    thr = best["threshold"]

    def probs(rows):
        return clf.predict_proba(vec.transform(rows))[:, 1]

    # ---- frozen regression suites (each once) ----
    te_ham_idx = [i for i in te if y[i] == 0]
    mod_texts = modern["test"] + [texts[i] for i in te_ham_idx]
    mod_y = np.concatenate([np.ones(len(modern["test"]), dtype=int),
                            np.zeros(len(te_ham_idx), dtype=int)])
    modern_m = metrics(mod_y, probs(mod_texts), thr)
    uci_m = metrics(y[te], probs(X(te)), thr)

    pf = probs([r["text"] for r in fam_test_all])
    fam_pred = (pf >= thr).astype(int)
    fam_overall = round(float(fam_pred.mean()), 4)
    per_family = {}
    for fam in sorted({r["family"] for r in fam_test_all}):
        sel = [i for i, r in enumerate(fam_test_all) if r["family"] == fam]
        per_family[fam] = {"recall": round(float(fam_pred[sel].mean()), 4),
                           "n": len(sel)}

    raw_texts = [augment_v4.demask(t, random.Random(f"demask-test-{i}"))
                 for i, t in enumerate(modern["test"])]
    v8_raw_recall = recall_block(probs(raw_texts), thr)

    txn_texts = [r["text"] for r in txn_test]
    txn_y = np.array([r["label"] for r in txn_test])
    p_txn = probs(txn_texts)
    pred_txn = (p_txn >= thr).astype(int)
    txn_block = {
        "threshold": round(float(thr), 4),
        "legit_fp_rate": round(float(pred_txn[txn_y == 0].mean()), 4),
        "legit_n": int((txn_y == 0).sum()),
        "legit_fp_count": int(pred_txn[txn_y == 0].sum()),
        "scam_recall": round(float(pred_txn[txn_y == 1].mean()), 4),
        "scam_n": int((txn_y == 1).sum()),
    }

    from probe_legit import PROBES
    p_probe = probs(PROBES)
    probe_block = {"n": len(PROBES),
                   "ge_threshold": int((p_probe >= thr).sum()),
                   "ge_0.70": int((p_probe >= 0.70).sum()),
                   "ge_0.30": int((p_probe >= 0.30).sum()),
                   "mean_prob": round(float(p_probe.mean()), 4)}

    # ---- NEW held-out sets (once; v8 and v5 baseline) ----
    def new_set(rows_scam, rows_legit=None):
        out = {}
        p8 = probs([r["text"] for r in rows_scam])
        p5 = np.array([predict_v5.prob_scam(r["text"]) for r in rows_scam])
        out["scam_recall_v8"] = recall_block(p8, thr)
        out["scam_recall_v5"] = recall_block(p5, predict_v5.THR)
        out["scam_n"] = len(rows_scam)
        per_t = {}
        for tname in sorted({r.get("transform") for r in rows_scam
                             if r.get("transform")}):
            sel = [i for i, r in enumerate(rows_scam)
                   if r.get("transform") == tname]
            per_t[tname] = {
                "recall_v8": recall_block(p8[sel], thr),
                "recall_v5": recall_block(p5[sel], predict_v5.THR),
                "n": len(sel)}
        if per_t:
            out["per_transform"] = per_t
        per_f = {}
        for fname in sorted({r.get("family") for r in rows_scam
                             if r.get("family")}):
            sel = [i for i, r in enumerate(rows_scam)
                   if r.get("family") == fname]
            per_f[fname] = {
                "recall_v8": recall_block(p8[sel], thr),
                "recall_v5": recall_block(p5[sel], predict_v5.THR),
                "n": len(sel)}
        if per_f:
            out["per_family"] = per_f
        if rows_legit is not None:
            q8 = probs([r["text"] for r in rows_legit])
            q5 = np.array([predict_v5.prob_scam(r["text"])
                           for r in rows_legit])
            out["legit_fp_rate_v8"] = round(
                float((q8 >= thr).mean()), 4)
            out["legit_fp_rate_v5"] = round(
                float((q5 >= predict_v5.THR).mean()), 4)
            out["legit_n"] = len(rows_legit)
        return out

    obf_block = new_set(obf_scam_eval, obf_legit_eval)
    hf_block = new_set(hf_eval)

    # ---- save candidate model + parity sample ----
    wv = dict(vec.transformer_list)["word"]
    cv = dict(vec.transformer_list)["char"]
    model_path = os.path.join(HERE, "model_v8.npz")
    np.savez_compressed(model_path,
                        word_vocab=np.asarray(wv.get_feature_names_out(),
                                              dtype="U64"),
                        word_idf=wv.idf_.astype(np.float32),
                        char_vocab=np.asarray(cv.get_feature_names_out(),
                                              dtype="U32"),
                        char_idf=cv.idf_.astype(np.float32),
                        coef=clf.coef_[0].astype(np.float32),
                        intercept=np.float32(clf.intercept_[0]),
                        threshold=np.float32(thr))
    size = os.path.getsize(model_path)
    with open(os.path.join(HERE, "val_parity_v8.json"), "w") as f:
        json.dump({"texts": val_texts[:200],
                   "probs": [round(float(p), 6) for p in pv[:200]]}, f)

    gates = {
        "modern_recall >= 0.965":
            bool(modern_m["scam_recall"] >= 0.965),
        "legit_fpr <= 0.005":
            bool(modern_m["legit_false_positive_rate"] <= 0.005),
        "txn_legit_fp <= 0.01":
            bool(txn_block["legit_fp_rate"] <= 0.01),
        "obf_C_recall > v5":
            bool(obf_block["scam_recall_v8"] > obf_block["scam_recall_v5"]),
        "heldout_family_recall >= v5":
            bool(hf_block["scam_recall_v8"] >= hf_block["scam_recall_v5"]),
    }
    keep = all(gates.values())
    report = {
        "model": "Scam Shield v8 — v7 recipe (word 1-2g 15k + char_wb "
                 "3-6g 35k TF-IDF, LR C=4.0, group-balanced weights, "
                 "set-A obfuscation copies, cross-family "
                 "recombinations, v7 hard-legit group kept) + threshold "
                 "chosen on validation with legit FPR cap 0.0040 "
                 "(midpoint) + NEW training families gig_task + "
                 "crypto_unlock (seeds_v8.json); romance + grandparent "
                 "held out as the fresh exam",
        "held_out_families": ["grandparent", "romance"],
        "val_fpr_cap": VAL_FPR_CAP,
        "val_cap_satisfied": bool(cap_satisfied),
        "validation_legit_n": val_legit_n,
        "train_rows": {f"group_{g}": int((groups == g).sum())
                       for g in range(len(group_texts))},
        "combined_train_rows": int(len(train_texts)),
        "validation_at_chosen_threshold": best,
        "test_modern_once": modern_m,
        "test_uci_once": uci_m,
        "test_family_once": {"threshold": round(float(thr), 4),
                             "scam_recall": fam_overall,
                             "n": len(fam_test_all),
                             "per_family": per_family},
        "raw_form_stress_test": {"n": len(raw_texts),
                                 "v8_recall_raw_form": v8_raw_recall,
                                 "v5_recall_raw_form": 0.9463,
                                 "v6_recall_raw_form": 0.9569,
                                 "v7_recall_raw_form": 0.9311},
        "test_transactional_once": {"v8": txn_block,
                                    "v5_reference": {
                                        "legit_fp_rate": 0.0035,
                                        "scam_recall": 1.0}},
        "probe_after": {"v8": probe_block,
                        "v5_reference": {"ge_threshold": 1, "ge_0.70": 1,
                                         "ge_0.30": 5, "mean_prob": 0.1053},
                        "v7_reference": {"ge_threshold": 0, "ge_0.70": 0,
                                         "ge_0.30": 3, "mean_prob": 0.0519}},
        "new_eval_obfuscated_C_once": obf_block,
        "new_eval_heldout_family_once": hf_block,
        "model_v8_npz_bytes": size,
        "keeper_gates": gates,
        "auto_keep_check": {
            "rule": "keep v8 iff ALL pre-registered gates pass: modern "
                    "recall >= 0.965, legit FPR <= 0.005, txn legit FP "
                    "<= 0.01, obf-C recall > v5 on set C, held-out-"
                    "family (romance+grandparent) recall >= v5",
            "keep_v8": bool(keep)},
        "comparison_table": [
            {"version": "v5", "threshold": 0.77,
             "modern_imc25_test_recall": 0.9674, "uci_test_recall": 0.9583,
             "legit_fpr": 0.0029, "txn_legit_fp_rate": 0.0035,
             "txn_scam_recall": 1.0, "model_bytes": 316854,
             "obf_C_recall": obf_block["scam_recall_v5"],
             "hf_romance_grandparent_recall": hf_block["scam_recall_v5"]},
            {"version": "v8", "threshold": round(float(thr), 4),
             "modern_imc25_test_recall": modern_m["scam_recall"],
             "uci_test_recall": uci_m["scam_recall"],
             "legit_fpr": modern_m["legit_false_positive_rate"],
             "txn_legit_fp_rate": txn_block["legit_fp_rate"],
             "txn_scam_recall": txn_block["scam_recall"],
             "model_bytes": size,
             "obf_C_recall": obf_block["scam_recall_v8"],
             "hf_romance_grandparent_recall": hf_block["scam_recall_v8"]},
        ],
    }
    with open(os.path.join(HERE, "metrics_v8.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("KEEP_V8:", bool(keep))
    print("GATES:", json.dumps(gates))
    print(json.dumps(report["comparison_table"], indent=2))
    print("OBF:", json.dumps(obf_block))
    print("HF:", json.dumps(hf_block))
    print("FAM task_job:", json.dumps(per_family.get("task_job")),
          "gig_task:", json.dumps(per_family.get("gig_task")),
          "crypto_unlock:", json.dumps(per_family.get("crypto_unlock")))
    print("SIZE:", size, "THR:", round(float(thr), 4),
          "VALCAP:", bool(cap_satisfied))


if __name__ == "__main__":
    main()
