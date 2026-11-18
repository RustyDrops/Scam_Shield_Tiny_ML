#!/usr/bin/env python3
"""Scam Shield v7: keep v6's pattern wins, fix the false-alarm rate.

v6 was rejected on ONE failure: test legit FPR 0.0059 vs the <=0.005
gate (its obfuscation gains were real: +1.6 pts on unseen transforms,
modern recall 0.9746). v7 keeps the v6 recipe and changes only the
FPR side, per the pre-registered design in PROVENANCE.md (v7 section,
written before this run):

1. Threshold chosen on VALIDATION ONLY, maximising scam recall subject
   to a HARD constraint: validation legit FPR <= 0.0030 (v6's cap was
   0.010; it landed at 0.0082 val / 0.0059 test). If no grid threshold
   satisfies the cap, the lowest-validation-FPR threshold is used and
   that is recorded.
2. A new group of hard legit training data (gen_v7_data.py): 1,186
   deterministic INFORM-ONLY transactional/casual texts (v5 style),
   plus a template-disjoint validation slice (189 rows) so the
   stricter threshold rule sees this text style.

Held-out discipline:
- toll + task_job families excluded from ALL training/validation
  (voter/jury_duty were v6's exam — spent, back in training).
- NEW obfuscation set B-v7 (gen_v7_data.py): base rows disjoint from
  v6's eval, fresh seeds, fresh set-B parameter variants + stacked
  pairs. Evaluated ONCE, for v7 and the v5 baseline.
- Held-out-family exam = toll + task_job fam_test rows, ONCE, both.
- Frozen v5 suites re-evaluated ONCE as regression.

Keeper gates (pre-registered, ALL must pass): modern recall >= 0.965;
legit FPR <= 0.005; transactional legit FP <= 0.01; obf-B-v7 recall
STRICTLY > v5's on set B-v7; held-out-family recall >= v5's on
toll + task_job.
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
HELD_OUT_FAMILIES = {"toll", "task_job"}
OBF_SCAM_COPIES = 4500
OBF_LEGIT_COPIES = 900
VAL_FPR_CAP = 0.0030


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

    fam_train_all = load_jsonl("data", "v4", "fam_train.jsonl")
    fam_val_all = load_jsonl("data", "v4", "fam_validation.jsonl")
    fam_test = load_jsonl("data", "v4", "fam_test.jsonl")
    fam_train = [r for r in fam_train_all
                 if r["family"] not in HELD_OUT_FAMILIES]
    fam_val = [r for r in fam_val_all
               if r["family"] not in HELD_OUT_FAMILIES]
    demasked_train = [r["text"] for r in
                      load_jsonl("data", "v4", "imc25_train_demasked.jsonl")]
    txn_train = load_jsonl("data", "v5", "txn_train.jsonl")
    txn_val = load_jsonl("data", "v5", "txn_validation.jsonl")
    txn_test = load_jsonl("data", "v5", "txn_test.jsonl")
    txn_train_legit = [r for r in txn_train if r["label"] == 0]
    txn_train_scam = [r for r in txn_train if r["label"] == 1]
    recomb_train = load_jsonl("data", "v7", "recomb_train.jsonl")
    recomb_val = load_jsonl("data", "v7", "recomb_validation.jsonl")
    obf_scam_eval = load_jsonl("data", "v7", "obf_eval_scam.jsonl")
    obf_legit_eval = load_jsonl("data", "v7", "obf_eval_legit.jsonl")
    hf_eval = load_jsonl("data", "v7", "hf_eval.jsonl")
    legit7_train = load_jsonl("data", "v7", "legit_train.jsonl")
    legit7_val = load_jsonl("data", "v7", "legit_validation.jsonl")

    uci_train_ham = [texts[i] for i in tr if y[i] == 0]
    uci_train_spam = [texts[i] for i in tr if y[i] == 1]

    # ---- obfuscated copies (set A transforms, seeded; v6 recipe) ----
    rng = random.Random("v7-obf-train")
    scam_pool = (modern["train"] + demasked_train
                 + [r["text"] for r in fam_train]
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
        [r["text"] for r in fam_train],                  # 4
        [r["text"] for r in txn_train_legit],            # 5 legit
        [r["text"] for r in txn_train_scam],             # 6
        [r["text"] for r in recomb_train],               # 7
        obf_scam,                                        # 8
        obf_legit,                                       # 9 legit
        [r["text"] for r in legit7_train],               # 10 legit (new)
    ]
    group_labels = [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
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
                 + [r["text"] for r in txn_val]
                 + [r["text"] for r in recomb_val]
                 + [r["text"] for r in legit7_val])
    val_y = np.concatenate([y[va],
                            np.ones(len(modern["validation"]), dtype=int),
                            np.ones(len(fam_val), dtype=int),
                            np.array([r["label"] for r in txn_val]),
                            np.ones(len(recomb_val), dtype=int),
                            np.zeros(len(legit7_val), dtype=int)])
    val_legit_n = int((val_y == 0).sum())

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

    pf = probs([r["text"] for r in fam_test])
    fam_pred = (pf >= thr).astype(int)
    fam_overall = round(float(fam_pred.mean()), 4)
    per_family = {}
    for fam in sorted({r["family"] for r in fam_test}):
        sel = [i for i, r in enumerate(fam_test) if r["family"] == fam]
        per_family[fam] = {"recall": round(float(fam_pred[sel].mean()), 4),
                           "n": len(sel)}

    raw_texts = [augment_v4.demask(t, random.Random(f"demask-test-{i}"))
                 for i, t in enumerate(modern["test"])]
    v7_raw_recall = recall_block(probs(raw_texts), thr)

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

    # ---- NEW held-out sets (once; v7 and v5 baseline) ----
    def new_set(rows_scam, rows_legit=None):
        out = {}
        p7 = probs([r["text"] for r in rows_scam])
        p5 = np.array([predict_v5.prob_scam(r["text"]) for r in rows_scam])
        out["scam_recall_v7"] = recall_block(p7, thr)
        out["scam_recall_v5"] = recall_block(p5, predict_v5.THR)
        out["scam_n"] = len(rows_scam)
        per_t = {}
        for tname in sorted({r.get("transform") for r in rows_scam
                             if r.get("transform")}):
            sel = [i for i, r in enumerate(rows_scam)
                   if r.get("transform") == tname]
            per_t[tname] = {
                "recall_v7": recall_block(p7[sel], thr),
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
                "recall_v7": recall_block(p7[sel], thr),
                "recall_v5": recall_block(p5[sel], predict_v5.THR),
                "n": len(sel)}
        if per_f:
            out["per_family"] = per_f
        if rows_legit is not None:
            q7 = probs([r["text"] for r in rows_legit])
            q5 = np.array([predict_v5.prob_scam(r["text"])
                           for r in rows_legit])
            out["legit_fp_rate_v7"] = round(
                float((q7 >= thr).mean()), 4)
            out["legit_fp_rate_v5"] = round(
                float((q5 >= predict_v5.THR).mean()), 4)
            out["legit_n"] = len(rows_legit)
        return out

    obf_block = new_set(obf_scam_eval, obf_legit_eval)
    hf_block = new_set(hf_eval)

    # ---- save candidate model + parity sample ----
    wv = dict(vec.transformer_list)["word"]
    cv = dict(vec.transformer_list)["char"]
    model_path = os.path.join(HERE, "model_v7.npz")
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
    with open(os.path.join(HERE, "val_parity_v7.json"), "w") as f:
        json.dump({"texts": val_texts[:200],
                   "probs": [round(float(p), 6) for p in pv[:200]]}, f)

    gates = {
        "modern_recall >= 0.965":
            bool(modern_m["scam_recall"] >= 0.965),
        "legit_fpr <= 0.005":
            bool(modern_m["legit_false_positive_rate"] <= 0.005),
        "txn_legit_fp <= 0.01":
            bool(txn_block["legit_fp_rate"] <= 0.01),
        "obf_B_v7_recall > v5":
            bool(obf_block["scam_recall_v7"] > obf_block["scam_recall_v5"]),
        "heldout_family_recall >= v5":
            bool(hf_block["scam_recall_v7"] >= hf_block["scam_recall_v5"]),
    }
    keep = all(gates.values())
    report = {
        "model": "Scam Shield v7 — v6 recipe (word 1-2g 15k + char_wb "
                 "3-6g 35k TF-IDF, LR C=4.0, group-balanced weights, "
                 "set-A obfuscation copies, cross-family recombinations)"
                 " + hard legit group (gen_v7_data) + threshold chosen "
                 "on validation with legit FPR cap 0.0030; toll and "
                 "task_job families held out",
        "held_out_families": sorted(HELD_OUT_FAMILIES),
        "val_fpr_cap": VAL_FPR_CAP,
        "val_cap_satisfied": bool(cap_satisfied),
        "validation_legit_n": val_legit_n,
        "train_rows": {f"group_{g}": int((groups == g).sum())
                       for g in range(len(group_texts))},
        "combined_train_rows": int(len(train_texts)),
        "validation_at_chosen_threshold": best,
        "test_modern_once": modern_m,
        "test_uci_once": uci_m,
        "test_family2026_once": {"threshold": round(float(thr), 4),
                                 "scam_recall": fam_overall,
                                 "n": len(fam_test),
                                 "per_family": per_family},
        "raw_form_stress_test": {"n": len(raw_texts),
                                 "v7_recall_raw_form": v7_raw_recall,
                                 "v5_recall_raw_form": 0.9463,
                                 "v6_recall_raw_form": 0.9569},
        "test_transactional_once": {"v7": txn_block,
                                    "v5_reference": {
                                        "legit_fp_rate": 0.0035,
                                        "scam_recall": 1.0}},
        "probe_after": {"v7": probe_block,
                        "v5_reference": {"ge_threshold": 1, "ge_0.70": 1,
                                         "ge_0.30": 5, "mean_prob": 0.1053}},
        "new_eval_obfuscated_B_v7_once": obf_block,
        "new_eval_heldout_family_once": hf_block,
        "model_v7_npz_bytes": size,
        "keeper_gates": gates,
        "auto_keep_check": {
            "rule": "keep v7 iff ALL pre-registered gates pass: modern "
                    "recall >= 0.965, legit FPR <= 0.005, txn legit FP "
                    "<= 0.01, obf-B-v7 recall > v5 on set B-v7, "
                    "held-out-family (toll+task_job) recall >= v5",
            "keep_v7": bool(keep)},
        "comparison_table": [
            {"version": "v5", "threshold": 0.77,
             "modern_imc25_test_recall": 0.9674, "uci_test_recall": 0.9583,
             "legit_fpr": 0.0029, "family2026_test_recall": 0.9758,
             "raw_form_recall": 0.9463, "txn_legit_fp_rate": 0.0035,
             "txn_scam_recall": 1.0, "model_bytes": 316854,
             "obf_B_v7_recall": obf_block["scam_recall_v5"],
             "hf_toll_taskjob_recall": hf_block["scam_recall_v5"]},
            {"version": "v7", "threshold": round(float(thr), 4),
             "modern_imc25_test_recall": modern_m["scam_recall"],
             "uci_test_recall": uci_m["scam_recall"],
             "legit_fpr": modern_m["legit_false_positive_rate"],
             "family2026_test_recall": fam_overall,
             "raw_form_recall": v7_raw_recall,
             "txn_legit_fp_rate": txn_block["legit_fp_rate"],
             "txn_scam_recall": txn_block["scam_recall"],
             "model_bytes": size,
             "obf_B_v7_recall": obf_block["scam_recall_v7"],
             "hf_toll_taskjob_recall": hf_block["scam_recall_v7"]},
        ],
    }
    with open(os.path.join(HERE, "metrics_v7.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("KEEP_V7:", bool(keep))
    print("GATES:", json.dumps(gates))
    print(json.dumps(report["comparison_table"], indent=2))
    print("OBF:", json.dumps(obf_block))
    print("HF:", json.dumps(hf_block))
    print("SIZE:", size, "THR:", round(float(thr), 4),
          "VALCAP:", bool(cap_satisfied))


if __name__ == "__main__":
    main()
