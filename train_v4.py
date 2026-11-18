#!/usr/bin/env python3
"""Scam Shield v4: v3 recipe + 2026-family coverage + de-masking fix.

What is new vs v3 (data only; model family and discipline unchanged):
- NEW positives: deterministic slot-fill augmentations of seed templates
  taken from US-government publications (FTC alerts/articles, FBI IC3 PSAs,
  IRS newsroom) — see seeds_v4.json and the v4 section of PROVENANCE.md.
  Split BY TEMPLATE: per family, template index 1 is validation-only and
  index 3 is test-only, so the 2026-family test measures generalisation to
  unseen seed templates, not memorised copies.
- NEW positives: a de-masked copy of every IMC25 train text (researcher
  placeholders <URL>/<PHONE_NUMBER>/... replaced by code-generated raw
  forms, augment_v4.demask) — targets v3's known weakness that masked
  placeholders are unrealistically easy features.
- Group-balanced sample weights extended to 5 groups (UCI ham, UCI spam,
  IMC25 masked, IMC25 de-masked, 2026-family) so no source drowns another.
- Same features (word 1-2g 15k + char_wb 3-5g 15k TF-IDF), same LR
  (C=4.0, balanced, liblinear), same threshold rule (validation only:
  max recall subject to legit FPR <= 1.0%).
- Tests evaluated ONCE: IMC25 test, UCI test, 2026-family held-out test.
- RAW-FORM stress test: IMC25 test positives de-masked programmatically;
  v3 and v4 recall compared on the identical raw-form rows.
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
from train_v2 import load_uci, load_modern, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def load_jsonl(name):
    path = os.path.join(HERE, "data", "v4", name)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


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

    fam_train = load_jsonl("fam_train.jsonl")
    fam_val = load_jsonl("fam_validation.jsonl")
    fam_test = load_jsonl("fam_test.jsonl")
    demasked_train = [r["text"] for r in load_jsonl("imc25_train_demasked.jsonl")]

    train_texts = (X(tr) + modern["train"] + demasked_train
                   + [r["text"] for r in fam_train])
    train_y = np.concatenate([
        y[tr],
        np.ones(len(modern["train"]), dtype=int),
        np.ones(len(demasked_train), dtype=int),
        np.ones(len(fam_train), dtype=int),
    ])
    groups = np.array(
        [0 if yy == 0 else 1 for yy in y[tr]]
        + [2] * len(modern["train"])
        + [3] * len(demasked_train)
        + [4] * len(fam_train))
    sample_w = np.zeros(len(groups))
    for g in range(5):
        sample_w[groups == g] = len(groups) / (5.0 * (groups == g).sum())

    val_texts = X(va) + modern["validation"] + [r["text"] for r in fam_val]
    val_y = np.concatenate([y[va],
                            np.ones(len(modern["validation"]), dtype=int),
                            np.ones(len(fam_val), dtype=int)])

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

    pv = clf.predict_proba(vec.transform(val_texts))[:, 1]
    best = None
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics(val_y, pv, thr)
        if m["legit_false_positive_rate"] <= 0.010:
            if best is None or m["scam_recall"] > best["scam_recall"]:
                best = m
    if best is None:
        f1s = []
        for thr in np.arange(0.05, 0.96, 0.01):
            m = metrics(val_y, pv, thr)
            p, r = m["precision_scam"], m["scam_recall"]
            f1s.append((2 * p * r / (p + r) if p + r else 0, m))
        best = max(f1s, key=lambda t: t[0])[1]
    thr = best["threshold"]

    pred_v = (pv >= thr).astype(int)
    n_uci_va, n_mod_va = len(va), len(modern["validation"])
    val_per_source = {
        "uci": round(float(pred_v[:n_uci_va][y[va] == 1].mean()), 4),
        "modern": round(float(pred_v[n_uci_va:n_uci_va + n_mod_va].mean()), 4),
        "family2026": round(float(pred_v[n_uci_va + n_mod_va:].mean()), 4),
    }

    # TEST (a) — modern IMC25 test positives vs UCI test ham, once
    te_ham_idx = [i for i in te if y[i] == 0]
    mod_texts = modern["test"] + [texts[i] for i in te_ham_idx]
    mod_y = np.concatenate([np.ones(len(modern["test"]), dtype=int),
                            np.zeros(len(te_ham_idx), dtype=int)])
    pm = clf.predict_proba(vec.transform(mod_texts))[:, 1]
    modern_m = metrics(mod_y, pm, thr)

    # TEST (b) — UCI full test split, once (identical rows to v1-v3)
    pt = clf.predict_proba(vec.transform(X(te)))[:, 1]
    uci_m = metrics(y[te], pt, thr)

    # TEST (c) — 2026-family held-out templates, once
    pf = clf.predict_proba(vec.transform([r["text"] for r in fam_test]))[:, 1]
    fam_pred = (pf >= thr).astype(int)
    fam_overall = round(float(fam_pred.mean()), 4)
    per_family = {}
    for fam in sorted({r["family"] for r in fam_test}):
        sel = [i for i, r in enumerate(fam_test) if r["family"] == fam]
        per_family[fam] = {"recall": round(float(fam_pred[sel].mean()), 4),
                           "n": len(sel)}
    fam_m = {"threshold": round(float(thr), 4), "scam_recall": fam_overall,
             "n": len(fam_test), "per_family": per_family}

    # RAW-FORM stress test — de-masked IMC25 test positives, v3 vs v4
    raw_texts = [augment_v4.demask(t, random.Random(f"demask-test-{i}"))
                 for i, t in enumerate(modern["test"])]
    p_raw_v4 = clf.predict_proba(vec.transform(raw_texts))[:, 1]
    v4_raw_recall = round(float(((p_raw_v4 >= thr)).mean()), 4)
    import predict_v3
    p_raw_v3 = np.array([predict_v3.prob_scam(t) for t in raw_texts])
    v3_raw_recall = round(float(((p_raw_v3 >= predict_v3.THR)).mean()), 4)
    raw_m = {"n": len(raw_texts), "v3_recall_raw_form": v3_raw_recall,
             "v4_recall_raw_form": v4_raw_recall,
             "v3_threshold": round(float(predict_v3.THR), 4),
             "v4_threshold": round(float(thr), 4),
             "note": "Identical de-masked rows for both models; de-masking "
                     "is deterministic code (augment_v4.demask), seeded per row."}

    wv = dict(vec.transformer_list)["word"]
    cv = dict(vec.transformer_list)["char"]
    model_path = os.path.join(HERE, "model_v4.npz")
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

    comparison = [
        {"version": "v1", "training": "UCI only", "threshold": 0.56,
         "modern_imc25_test_recall": 0.4516, "uci_test_recall": 0.9167,
         "legit_fpr": 0.0015, "family2026_test_recall": None,
         "raw_form_recall": None, "model_bytes": 57541},
        {"version": "v2", "training": "UCI train + IMC25 positives (unweighted)",
         "threshold": 0.72, "modern_imc25_test_recall": 0.9569,
         "uci_test_recall": 0.7604, "legit_fpr": 0.0044,
         "family2026_test_recall": None, "raw_form_recall": None,
         "model_bytes": 62670},
        {"version": "v3", "training": "UCI train + IMC25 positives, group-balanced weights, word+char TF-IDF 15k+15k",
         "threshold": 0.66, "modern_imc25_test_recall": 0.9742,
         "uci_test_recall": 0.9479, "legit_fpr": 0.0029,
         "family2026_test_recall": None,
         "raw_form_recall": v3_raw_recall, "model_bytes": 310698},
        {"version": "v4", "training": "v3 data + IMC25 de-masked copies + 2026-family seed augmentations (gov sources), 5-group balanced",
         "threshold": thr,
         "modern_imc25_test_recall": modern_m["scam_recall"],
         "uci_test_recall": uci_m["scam_recall"],
         "legit_fpr": modern_m["legit_false_positive_rate"],
         "family2026_test_recall": fam_overall,
         "raw_form_recall": v4_raw_recall, "model_bytes": size},
    ]
    keep_v4 = (modern_m["scam_recall"] >= 0.9742 - 0.01
               and uci_m["scam_recall"] >= 0.9479 - 0.01
               and modern_m["legit_false_positive_rate"] <= 0.0029 + 0.005)
    report = {
        "model": "Scam Shield v4 — TF-IDF (word 1-2g 15k + char_wb 3-5g 15k) + Logistic Regression, 5-group balanced sample weights",
        "training_data": "UCI train split + IMC25 train positives (masked originals AND deterministic de-masked copies) + 2026-family augmentations of government-published seed templates (seeds_v4.json)",
        "new_data_counts": {
            "family2026_train_rows": len(fam_train),
            "family2026_validation_rows": len(fam_val),
            "family2026_test_rows": len(fam_test),
            "imc25_demasked_train_rows": len(demasked_train),
        },
        "modern_dataset_stats": mstats,
        "combined_split_sizes": {"train": int(len(train_texts)),
                                 "validation": int(len(val_texts))},
        "validation_at_chosen_threshold": best,
        "validation_per_source_recall": val_per_source,
        "test_modern_once": modern_m,
        "test_uci_once": uci_m,
        "test_family2026_once": fam_m,
        "raw_form_stress_test": raw_m,
        "comparison_table": comparison,
        "model_v4_npz_bytes": size,
        "vocab_size": int(len(wv.get_feature_names_out())
                          + len(cv.get_feature_names_out())),
        "auto_keep_check": {
            "rule": "keep v4 if modern and UCI test recall within 1pt of v3 and legit FPR no more than 0.5pt worse",
            "keep_v4": bool(keep_v4),
        },
    }
    with open(os.path.join(HERE, "metrics_v4.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
