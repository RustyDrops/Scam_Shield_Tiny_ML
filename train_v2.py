#!/usr/bin/env python3
"""Scam Shield v2: TF-IDF + Logistic Regression on UCI SMS Spam (v1 splits,
reproduced exactly) PLUS modern smishing positives from the IMC 2025 dataset
(Agarwal et al., user-reported smishing, CC BY 4.0), English subset only.

Design (pre-registered):
- UCI load/dedup/split identical to train.py (seed 42, 70/15/15 stratified),
  so the v1 test split is byte-identical and the v1 comparison is clean.
- Modern positives: IMC25 English rows, exact-deduped, email-screened,
  >=15 chars, split 70/15/15 by seed-42 shuffle. Any modern text that also
  appears in the UCI validation or test split is DROPPED (leakage guard).
- Same model family/hyperparameters as v1: TF-IDF word 1-2 grams,
  max 5,000 features, min_df 2, sublinear tf; Logistic Regression C=4.0,
  class_weight=balanced, liblinear.
- Threshold: ONLY on combined validation (UCI val + modern val positives),
  same fixed rule as v1: highest scam recall subject to legit FPR <= 1.0%.
- Test evaluated once at that threshold, reported two ways:
  (a) modern test = modern test positives vs UCI test ham (held-out legit);
  (b) v1 comparison = full UCI test split, same rows v1 was scored on.
"""
import json, zipfile, csv, re, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "data", "sms.zip")
IMC = os.path.join(HERE, "data", "modern", "imc25.csv")
SEED = 42
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

def load_uci():
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
    texts, labels, seen = [], [], {}
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        lab, txt = line.split("\t", 1)
        lab = 1 if lab.strip() == "spam" else 0
        key = txt.strip().lower()
        if key in seen:
            continue
        seen[key] = lab
        texts.append(txt); labels.append(lab)
    return texts, np.array(labels)

def load_modern(uci_val_test_keys):
    rows = list(csv.DictReader(open(IMC, encoding="utf-8", errors="replace")))
    stats = {"rows_total": len(rows), "english": 0, "dropped_email": 0,
             "dropped_short": 0, "dropped_uci_valtest_overlap": 0, "kept": 0}
    texts, seen = [], set()
    for r in rows:
        if r["language"] != "English":
            continue
        stats["english"] += 1
        t = (r["text"] or "").strip()
        if len(t) < 15:
            stats["dropped_short"] += 1; continue
        if EMAIL_RE.search(t):
            stats["dropped_email"] += 1; continue
        key = t.lower()
        if key in seen:
            continue
        if key in uci_val_test_keys:
            stats["dropped_uci_valtest_overlap"] += 1; continue
        seen.add(key)
        texts.append(t)
    stats["kept"] = len(texts)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(texts))
    n = len(texts)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    splits = {
        "train": [texts[i] for i in perm[:n_tr]],
        "validation": [texts[i] for i in perm[n_tr:n_tr + n_va]],
        "test": [texts[i] for i in perm[n_tr + n_va:]],
    }
    return splits, stats

def metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 4),
        "scam_recall": round(float(tp / (tp + fn)) if tp + fn else 0.0, 4),
        "legit_false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 4),
        "precision_scam": round(float(tp / (tp + fp)) if tp + fp else 0.0, 4),
        "accuracy": round(float((tp + tn) / len(y)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "n": int(len(y)),
    }

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
    val_texts = X(va) + modern["validation"]
    val_y = np.concatenate([y[va], np.ones(len(modern["validation"]), dtype=int)])

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000,
                          min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']+\b")
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000, solver="liblinear")
    clf.fit(Xtr, train_y)

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

    # TEST (a) — modern: modern test positives vs UCI test ham (held-out legit)
    te_ham_idx = [i for i in te if y[i] == 0]
    mod_texts = modern["test"] + [texts[i] for i in te_ham_idx]
    mod_y = np.concatenate([np.ones(len(modern["test"]), dtype=int),
                            np.zeros(len(te_ham_idx), dtype=int)])
    pm = clf.predict_proba(vec.transform(mod_texts))[:, 1]
    modern_m = metrics(mod_y, pm, thr)

    # TEST (b) — v1 comparison: full UCI test split, identical rows to v1
    pt = clf.predict_proba(vec.transform(X(te)))[:, 1]
    uci_m = metrics(y[te], pt, thr)

    vocab = vec.get_feature_names_out()
    model_path = os.path.join(HERE, "model_v2.npz")
    np.savez_compressed(model_path, vocab=np.asarray(vocab, dtype="U64"),
                        idf=vec.idf_.astype(np.float32),
                        coef=clf.coef_[0].astype(np.float32),
                        intercept=np.float32(clf.intercept_[0]),
                        threshold=np.float32(thr))

    report = {
        "model": "Scam Shield v2 — TF-IDF + Logistic Regression (same recipe as v1)",
        "training_data": "UCI SMS Spam Collection train split (v1, identical) + IMC25 English smishing positives (train split)",
        "modern_dataset_stats": mstats,
        "modern_split_sizes": {k: len(v) for k, v in modern.items()},
        "combined_split_sizes": {"train": int(len(train_texts)), "validation": int(len(val_texts))},
        "validation_at_chosen_threshold": best,
        "test_modern_once": modern_m,
        "test_uci_v1_comparison_once": uci_m,
        "v1_reference": {"threshold": 0.56, "uci_test_scam_recall": 0.9167,
                         "uci_test_legit_fpr": 0.0015, "model_npz_bytes": 57541},
        "model_v2_npz_bytes": os.path.getsize(model_path),
        "vocab_size": int(len(vocab)),
    }
    with open(os.path.join(HERE, "metrics_v2.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
