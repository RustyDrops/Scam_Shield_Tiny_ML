#!/usr/bin/env python3
"""Scam Shield v3: combined UCI + IMC25 training, recipe selected on
VALIDATION ONLY (sweep in /tmp during development; test touched once here).

What changed vs v2 (and why, from validation evidence):
- v2 let 11,033 modern positives drown 449 UCI train spam; UCI validation
  recall was 0.8041 at v2's rule. v3 uses GROUP-BALANCED sample weights:
  UCI ham, UCI spam, and IMC25 positives each contribute equal total weight.
- Features: word 1-2 grams (15k) UNION char_wb 3-5 grams (15k), TF-IDF
  sublinear, min_df 2. On validation this gave min(UCI, modern) recall
  0.9278 (UCI 0.9278 / modern 0.9632) vs v2 recipe's 0.8041 / 0.9509.
- Same splits as v1/v2 (seed 42), same LR (C=4.0, balanced, liblinear),
  same threshold rule: max recall on combined validation subject to
  legit FPR <= 1.0%. Datasets identical to v2: UCI (CC BY 4.0) train
  ham+spam plus IMC25 (CC BY 4.0) English smishing positives; legitimate
  negatives are UCI ham (IMC25 has no legitimate messages).
- Tests evaluated ONCE at the chosen threshold: (a) modern IMC25 test
  positives vs UCI test ham; (b) full UCI test split (v1 comparison rows).
"""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from train_v2 import load_uci, load_modern, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42

def main():
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y[tmp])
    X = lambda ii: [texts[i] for i in ii]
    uci_val_test_keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, mstats = load_modern(uci_val_test_keys)

    train_texts = X(tr) + modern["train"]
    train_y = np.concatenate([y[tr], np.ones(len(modern["train"]), dtype=int)])
    groups = np.array([0 if yy == 0 else 1 for yy in y[tr]] + [2] * len(modern["train"]))
    sample_w = np.zeros(len(groups))
    for g in range(3):
        sample_w[groups == g] = len(groups) / (3.0 * (groups == g).sum())

    val_texts = X(va) + modern["validation"]
    val_y = np.concatenate([y[va], np.ones(len(modern["validation"]), dtype=int)])

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=15000,
                           min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']+\b")
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                           max_features=15000, min_df=2, sublinear_tf=True)
    vec = FeatureUnion([("word", word), ("char", char)])
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000, solver="liblinear")
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

    # Per-source validation recalls at chosen threshold (transparency)
    pred_v = (pv >= thr).astype(int)
    uci_val_pos = pred_v[:len(va)][y[va] == 1].mean()
    mod_val_pos = pred_v[len(va):].mean()

    # TEST (a) — modern: IMC25 test positives vs UCI test ham, once
    te_ham_idx = [i for i in te if y[i] == 0]
    mod_texts = modern["test"] + [texts[i] for i in te_ham_idx]
    mod_y = np.concatenate([np.ones(len(modern["test"]), dtype=int),
                            np.zeros(len(te_ham_idx), dtype=int)])
    pm = clf.predict_proba(vec.transform(mod_texts))[:, 1]
    modern_m = metrics(mod_y, pm, thr)

    # TEST (b) — UCI full test split, once (identical rows to v1/v2)
    pt = clf.predict_proba(vec.transform(X(te)))[:, 1]
    uci_m = metrics(y[te], pt, thr)

    wv = dict(vec.transformer_list)["word"]; cv = dict(vec.transformer_list)["char"]
    n_word = len(wv.get_feature_names_out())
    model_path = os.path.join(HERE, "model_v3.npz")
    np.savez_compressed(model_path,
                        word_vocab=np.asarray(wv.get_feature_names_out(), dtype="U64"),
                        word_idf=wv.idf_.astype(np.float32),
                        char_vocab=np.asarray(cv.get_feature_names_out(), dtype="U32"),
                        char_idf=cv.idf_.astype(np.float32),
                        coef=clf.coef_[0].astype(np.float32),
                        intercept=np.float32(clf.intercept_[0]),
                        threshold=np.float32(thr))
    size = os.path.getsize(model_path)

    comparison = [
        {"version": "v1", "training": "UCI only", "threshold": 0.56,
         "modern_imc25_test_recall": 0.4516, "uci_test_recall": 0.9167,
         "legit_fpr": 0.0015, "model_bytes": 57541},
        {"version": "v2", "training": "UCI train + IMC25 positives (unweighted)", "threshold": 0.72,
         "modern_imc25_test_recall": 0.9569, "uci_test_recall": 0.7604,
         "legit_fpr": 0.0044, "model_bytes": 62670},
        {"version": "v3", "training": "UCI train + IMC25 positives, group-balanced weights, word+char TF-IDF 15k+15k",
         "threshold": thr,
         "modern_imc25_test_recall": modern_m["scam_recall"], "uci_test_recall": uci_m["scam_recall"],
         "legit_fpr": modern_m["legit_false_positive_rate"], "model_bytes": size},
    ]
    report = {
        "model": "Scam Shield v3 — TF-IDF (word 1-2g 15k + char_wb 3-5g 15k) + Logistic Regression, group-balanced sample weights",
        "training_data": "UCI SMS Spam Collection train split (ham+spam) + IMC25 English smishing positives (train split); legitimate negatives are UCI ham only (IMC25 contains no legitimate messages)",
        "recipe_selection": "Chosen on validation only: maximize min(UCI, modern) validation recall, tie-break combined validation recall. Validation at chosen threshold: UCI recall %.4f, modern recall %.4f" % (uci_val_pos, mod_val_pos),
        "modern_dataset_stats": mstats,
        "modern_split_sizes": {k: len(v) for k, v in modern.items()},
        "combined_split_sizes": {"train": int(len(train_texts)), "validation": int(len(val_texts))},
        "validation_at_chosen_threshold": best,
        "validation_per_source_recall": {"uci": round(float(uci_val_pos), 4), "modern": round(float(mod_val_pos), 4)},
        "test_modern_once": modern_m,
        "test_uci_once": uci_m,
        "comparison_table": comparison,
        "model_v3_npz_bytes": size,
        "vocab_size": int(n_word + len(cv.get_feature_names_out())),
    }
    with open(os.path.join(HERE, "metrics_v3.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
