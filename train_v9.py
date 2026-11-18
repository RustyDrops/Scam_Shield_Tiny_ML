#!/usr/bin/env python3
"""Scam Shield v9: v5 recipe + conversational-scam data.

Pre-registered in PROVENANCE.md (v9 section) BEFORE any v9 training.

v6-v8 lessons: threshold-cap games are exhausted (v7 cap 0.30% ->
thr 0.78, v8 cap 0.40% -> thr 0.81; modern recall fell 0.9674 ->
0.9619 -> 0.9543), and v8's fresh exam exposed a DATA gap:
conversational/emotional scams (romance, grandparent) scored
0.14-0.21 for BOTH v5 and v8 — few URLs/keywords for TF-IDF to grip.

v9 = v5 EXACTLY (feature config: word 1-2g 15k + char_wb 3-5g 15k
TF-IDF, LR C=4.0 balanced liblinear; v5's seven training groups;
threshold by v5's own rule: max recall s.t. validation legit FPR
<= 1.0%) + ADDED data only, as two new balanced groups:
  group 7: conversational scam — romance + grandparent (seeds_v8,
           v4 template split) + wrong_number_opener (seeds_v9 NEW)
  group 8: conversational legit — gen_v9_data templates (genuine
           wrong-number replies, wrong-person apologies, family /
           friendly chat), template-disjoint validation slice
Safety valve (recorded if used): if no grid threshold satisfies
the 1.0% rule, fall back to the lowest-validation-FPR threshold.

Fresh evals, built before training (gen_v9_data.py), each ONCE,
v5 scored once on each as baseline:
  - held-out families charity_relief + package_delivery (seeds_v9;
    charity never used in any round; package_delivery new as a
    seed family, delivery register partially seen — disclosed)
  - obfuscation set D: bases from the VALIDATION pools (the test
    pools are exhausted by sets v6/B-v7/C), OBF_B7 transforms,
    fresh v9 seeds
  - fam9 internal test splits (romance/grandparent/wrong_number_
    opener template index 3) — informational, reported by name
Frozen v5 suites (modern, UCI, 2026-family, raw-form stress,
transactional, probe) re-evaluated once as regression.

Keeper gates (pre-registered, ALL must pass): modern recall >=
0.965; legit FPR <= 0.005; transactional legit FP <= 0.01; obf-D
recall STRICTLY > v5's on set D; held-out-family (charity_relief
+ package_delivery) recall >= v5's on the same families.
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
import predict_v5
from train_v2 import load_uci, load_modern, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
VAL_FPR_RULE = 0.010


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

    # ---- v5's exact training data ----
    fam_train = load_jsonl("data", "v4", "fam_train.jsonl")
    fam_val = load_jsonl("data", "v4", "fam_validation.jsonl")
    fam_test = load_jsonl("data", "v4", "fam_test.jsonl")
    demasked_train = [r["text"] for r in
                      load_jsonl("data", "v4", "imc25_train_demasked.jsonl")]
    txn_train = load_jsonl("data", "v5", "txn_train.jsonl")
    txn_val = load_jsonl("data", "v5", "txn_validation.jsonl")
    txn_test = load_jsonl("data", "v5", "txn_test.jsonl")
    txn_train_legit = [r for r in txn_train if r["label"] == 0]
    txn_train_scam = [r for r in txn_train if r["label"] == 1]

    # ---- v9 added data ----
    fam9_train = load_jsonl("data", "v9", "fam9_train.jsonl")
    fam9_val = load_jsonl("data", "v9", "fam9_validation.jsonl")
    fam9_test = load_jsonl("data", "v9", "fam9_test.jsonl")
    legit9_train = load_jsonl("data", "v9", "legit_conv_train.jsonl")
    legit9_val = load_jsonl("data", "v9", "legit_conv_validation.jsonl")
    obf_scam_eval = load_jsonl("data", "v9", "obf_eval_scam.jsonl")
    obf_legit_eval = load_jsonl("data", "v9", "obf_eval_legit.jsonl")
    hf_eval = load_jsonl("data", "v9", "hf_eval.jsonl")

    uci_train_ham = [texts[i] for i in tr if y[i] == 0]
    uci_train_spam = [texts[i] for i in tr if y[i] == 1]

    group_texts = [
        uci_train_ham,                                   # 0 legit (v5)
        uci_train_spam,                                  # 1 (v5)
        modern["train"],                                 # 2 (v5)
        demasked_train,                                  # 3 (v5)
        [r["text"] for r in fam_train],                  # 4 (v5)
        [r["text"] for r in txn_train_legit],            # 5 legit (v5)
        [r["text"] for r in txn_train_scam],             # 6 (v5)
        [r["text"] for r in fam9_train],                 # 7 NEW conv scam
        [r["text"] for r in legit9_train],               # 8 NEW conv legit
    ]
    group_labels = [0, 1, 1, 1, 1, 0, 1, 1, 0]
    train_texts, train_y_list, groups = [], [], []
    for g, (gt, lab) in enumerate(zip(group_texts, group_labels)):
        train_texts += gt
        train_y_list += [lab] * len(gt)
        groups += [g] * len(gt)
    train_y = np.array(train_y_list)
    groups = np.array(groups)
    n_groups = len(group_texts)
    sample_w = np.zeros(len(groups))
    for g in range(n_groups):
        m = groups == g
        sample_w[m] = len(groups) / (float(n_groups) * m.sum())

    val_texts = (X(va) + modern["validation"]
                 + [r["text"] for r in fam_val]
                 + [r["text"] for r in txn_val]
                 + [r["text"] for r in fam9_val]
                 + [r["text"] for r in legit9_val])
    val_y = np.concatenate([y[va],
                            np.ones(len(modern["validation"]), dtype=int),
                            np.ones(len(fam_val), dtype=int),
                            np.array([r["label"] for r in txn_val]),
                            np.ones(len(fam9_val), dtype=int),
                            np.zeros(len(legit9_val), dtype=int)])
    val_legit_n = int((val_y == 0).sum())

    # ---- leakage asserts (exam / set-D material not in train/val) ----
    train_keys = {t.strip().lower() for t in train_texts}
    val_keys = {t.strip().lower() for t in val_texts}
    for name, rows in (("hf_eval", hf_eval),
                       ("obf_eval_scam", obf_scam_eval),
                       ("obf_eval_legit", obf_legit_eval),
                       ("fam9_test", fam9_test)):
        for r in rows:
            k = r["text"].strip().lower()
            assert k not in train_keys, f"{name} row leaked into train"
            assert k not in val_keys, f"{name} row leaked into val"
    print("LEAKAGE_ASSERTS_PASSED", flush=True)

    # ---- v5's exact feature config ----
    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                           max_features=15000, min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w[\w']+\b")
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb",
                           ngram_range=(3, 5), max_features=15000, min_df=2,
                           sublinear_tf=True)
    vec = FeatureUnion([("word", word), ("char", char)])
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000,
                             solver="liblinear")
    clf.fit(Xtr, train_y, sample_weight=sample_w)

    # ---- threshold: v5's own rule (max recall s.t. val FPR <= 1%) ----
    pv = clf.predict_proba(vec.transform(val_texts))[:, 1]
    best = None
    lowest_fpr = None
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics(val_y, pv, thr)
        if lowest_fpr is None or \
                m["legit_false_positive_rate"] < \
                lowest_fpr["legit_false_positive_rate"]:
            lowest_fpr = m
        if m["legit_false_positive_rate"] <= VAL_FPR_RULE:
            if best is None or m["scam_recall"] > best["scam_recall"]:
                best = m
    rule_satisfied = best is not None
    if not rule_satisfied:
        best = lowest_fpr
    thr = best["threshold"]

    def probs(rows):
        return clf.predict_proba(vec.transform(rows))[:, 1]

    def probs5(rows):
        return np.array([predict_v5.prob_scam(t) for t in rows])

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
    v9_raw_recall = recall_block(probs(raw_texts), thr)
    v5_raw_recall = recall_block(probs5(raw_texts), predict_v5.THR)

    txn_texts = [r["text"] for r in txn_test]
    txn_y = np.array([r["label"] for r in txn_test])

    def txn_block(p, t):
        pred = (p >= t).astype(int)
        return {"threshold": round(float(t), 4),
                "legit_fp_rate": round(float(pred[txn_y == 0].mean()), 4),
                "legit_n": int((txn_y == 0).sum()),
                "legit_fp_count": int(pred[txn_y == 0].sum()),
                "scam_recall": round(float(pred[txn_y == 1].mean()), 4),
                "scam_n": int((txn_y == 1).sum())}
    txn_v9 = txn_block(probs(txn_texts), thr)
    txn_v5 = txn_block(probs5(txn_texts), predict_v5.THR)

    from probe_legit import PROBES

    def probe_block(p, t):
        return {"n": len(PROBES),
                "ge_threshold": int((p >= t).sum()),
                "ge_0.70": int((p >= 0.70).sum()),
                "ge_0.30": int((p >= 0.30).sum()),
                "mean_prob": round(float(p.mean()), 4)}
    probe = {"v9": probe_block(probs(PROBES), thr),
             "v5": probe_block(probs5(PROBES), predict_v5.THR)}

    # ---- fam9 internal test splits (informational; v9 and v5) ----
    p9_fam9 = probs([r["text"] for r in fam9_test])
    p5_fam9 = probs5([r["text"] for r in fam9_test])
    fam9_block = {}
    for fam in sorted({r["family"] for r in fam9_test}):
        sel = [i for i, r in enumerate(fam9_test) if r["family"] == fam]
        fam9_block[fam] = {
            "recall_v9": recall_block(p9_fam9[sel], thr),
            "recall_v5": recall_block(p5_fam9[sel], predict_v5.THR),
            "n": len(sel)}

    # ---- fresh held-out sets (once; v9 and v5 baseline) ----
    def new_set(rows_scam, rows_legit=None):
        out = {}
        p9 = probs([r["text"] for r in rows_scam])
        p5 = probs5([r["text"] for r in rows_scam])
        out["scam_recall_v9"] = recall_block(p9, thr)
        out["scam_recall_v5"] = recall_block(p5, predict_v5.THR)
        out["scam_n"] = len(rows_scam)
        per_t = {}
        for tname in sorted({r.get("transform") for r in rows_scam
                             if r.get("transform")}):
            sel = [i for i, r in enumerate(rows_scam)
                   if r.get("transform") == tname]
            per_t[tname] = {
                "recall_v9": recall_block(p9[sel], thr),
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
                "recall_v9": recall_block(p9[sel], thr),
                "recall_v5": recall_block(p5[sel], predict_v5.THR),
                "n": len(sel)}
        if per_f:
            out["per_family"] = per_f
        if rows_legit is not None:
            q9 = probs([r["text"] for r in rows_legit])
            q5 = probs5([r["text"] for r in rows_legit])
            out["legit_fp_rate_v9"] = round(
                float((q9 >= thr).mean()), 4)
            out["legit_fp_rate_v5"] = round(
                float((q5 >= predict_v5.THR).mean()), 4)
            out["legit_n"] = len(rows_legit)
        return out

    obf_block = new_set(obf_scam_eval, obf_legit_eval)
    hf_block = new_set(hf_eval)

    # ---- save candidate model (v5 npz format) + parity sample ----
    wv = dict(vec.transformer_list)["word"]
    cv = dict(vec.transformer_list)["char"]
    model_path = os.path.join(HERE, "model_v9.npz")
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
    with open(os.path.join(HERE, "val_parity_v9.json"), "w") as f:
        json.dump({"texts": val_texts[:200],
                   "probs": [round(float(p), 6) for p in pv[:200]]}, f)

    gates = {
        "modern_recall >= 0.965":
            bool(modern_m["scam_recall"] >= 0.965),
        "legit_fpr <= 0.005":
            bool(modern_m["legit_false_positive_rate"] <= 0.005),
        "txn_legit_fp <= 0.01":
            bool(txn_v9["legit_fp_rate"] <= 0.01),
        "obf_D_recall > v5":
            bool(obf_block["scam_recall_v9"] > obf_block["scam_recall_v5"]),
        "heldout_family_recall >= v5":
            bool(hf_block["scam_recall_v9"] >= hf_block["scam_recall_v5"]),
    }
    keep = all(gates.values())
    report = {
        "model": "Scam Shield v9 — v5's exact recipe (word 1-2g 15k + "
                 "char_wb 3-5g 15k TF-IDF, LR C=4.0, group-balanced "
                 "weights, threshold by v5's rule: max recall s.t. "
                 "validation legit FPR <= 1.0%) + ADDED data only: "
                 "conversational scam group (romance + grandparent "
                 "from seeds_v8, wrong_number_opener NEW in "
                 "seeds_v9) and conversational legit group "
                 "(gen_v9_data templates); charity_relief + "
                 "package_delivery held out as the fresh exam",
        "held_out_families": ["charity_relief", "package_delivery"],
        "threshold_rule": "v5 rule: max recall s.t. validation legit "
                          "FPR <= 0.010 (no cap variant)",
        "threshold_rule_satisfied": bool(rule_satisfied),
        "validation_legit_n": val_legit_n,
        "train_rows": {f"group_{g}": int((groups == g).sum())
                       for g in range(n_groups)},
        "combined_train_rows": int(len(train_texts)),
        "validation_at_chosen_threshold": best,
        "test_modern_once": modern_m,
        "test_uci_once": uci_m,
        "test_family2026_once": {"threshold": round(float(thr), 4),
                                 "scam_recall": fam_overall,
                                 "n": len(fam_test),
                                 "per_family": per_family},
        "raw_form_stress_test": {"n": len(raw_texts),
                                 "v9_recall_raw_form": v9_raw_recall,
                                 "v5_recall_raw_form": v5_raw_recall},
        "test_transactional_once": {"v9": txn_v9, "v5_same_rows": txn_v5},
        "probe_after": probe,
        "fam9_internal_test_once": fam9_block,
        "new_eval_obfuscated_D_once": obf_block,
        "new_eval_heldout_family_once": hf_block,
        "model_v9_npz_bytes": size,
        "keeper_gates": gates,
        "auto_keep_check": {
            "rule": "keep v9 iff ALL pre-registered gates pass: modern "
                    "recall >= 0.965, legit FPR <= 0.005, txn legit FP "
                    "<= 0.01, obf-D recall > v5 on set D, held-out-"
                    "family (charity_relief+package_delivery) recall "
                    ">= v5",
            "keep_v9": bool(keep)},
        "comparison_table": [
            {"version": "v5", "threshold": 0.77,
             "modern_imc25_test_recall": 0.9674, "uci_test_recall": 0.9583,
             "legit_fpr": 0.0029, "txn_legit_fp_rate": 0.0035,
             "txn_scam_recall": 1.0, "model_bytes": 316854,
             "obf_D_recall": obf_block["scam_recall_v5"],
             "hf_charity_package_recall": hf_block["scam_recall_v5"],
             "fam9_romance_recall":
                 fam9_block.get("romance", {}).get("recall_v5"),
             "fam9_grandparent_recall":
                 fam9_block.get("grandparent", {}).get("recall_v5")},
            {"version": "v9", "threshold": round(float(thr), 4),
             "modern_imc25_test_recall": modern_m["scam_recall"],
             "uci_test_recall": uci_m["scam_recall"],
             "legit_fpr": modern_m["legit_false_positive_rate"],
             "txn_legit_fp_rate": txn_v9["legit_fp_rate"],
             "txn_scam_recall": txn_v9["scam_recall"],
             "model_bytes": size,
             "obf_D_recall": obf_block["scam_recall_v9"],
             "hf_charity_package_recall": hf_block["scam_recall_v9"],
             "fam9_romance_recall":
                 fam9_block.get("romance", {}).get("recall_v9"),
             "fam9_grandparent_recall":
                 fam9_block.get("grandparent", {}).get("recall_v9")},
        ],
    }
    with open(os.path.join(HERE, "metrics_v9.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("KEEP_V9:", bool(keep))
    print("GATES:", json.dumps(gates))
    print(json.dumps(report["comparison_table"], indent=2))
    print("OBF_D:", json.dumps(obf_block))
    print("HF:", json.dumps(hf_block))
    print("FAM9:", json.dumps(fam9_block))
    print("SIZE:", size, "THR:", round(float(thr), 4),
          "RULE_SATISFIED:", bool(rule_satisfied))


if __name__ == "__main__":
    main()
