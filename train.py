#!/usr/bin/env python3
"""Scam Shield baseline: TF-IDF + Logistic Regression on UCI SMS Spam Collection.

Splits: train 70% / validation 15% / test 15%, stratified, seed 42.
Decision threshold is tuned ONLY on validation by a fixed script rule:
among thresholds giving validation FPR <= 1.0% on legitimate (ham) messages,
pick the one with highest scam (spam) recall; if none qualifies, pick the
threshold maximizing validation F1. Test set is evaluated exactly once,
at that threshold.
"""
import json, zipfile, io, hashlib, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "data", "sms.zip")
SEED = 42

def load():
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
    texts, labels, seen = [], [], {}
    conflicts = 0
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        lab, txt = line.split("\t", 1)
        lab = 1 if lab.strip() == "spam" else 0
        key = txt.strip().lower()
        if key in seen:
            if seen[key] != lab:
                conflicts += 1
            continue
        seen[key] = lab
        texts.append(txt); labels.append(lab)
    return texts, np.array(labels), conflicts

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
    texts, y, conflicts = load()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y[tmp])
    X = lambda ii: [texts[i] for i in ii]

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000,
                          min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']+\b")
    Xtr = vec.fit_transform(X(tr))
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000, solver="liblinear")
    clf.fit(Xtr, y[tr])

    pv = clf.predict_proba(vec.transform(X(va)))[:, 1]
    # Fixed threshold rule on validation only
    best = None
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics(y[va], pv, thr)
        if m["legit_false_positive_rate"] <= 0.010:
            if best is None or m["scam_recall"] > best["scam_recall"]:
                best = m
    if best is None:  # fallback: max F1 on validation
        f1s = []
        for thr in np.arange(0.05, 0.96, 0.01):
            m = metrics(y[va], pv, thr)
            p, r = m["precision_scam"], m["scam_recall"]
            f1s.append((2 * p * r / (p + r) if p + r else 0, m))
        best = max(f1s, key=lambda t: t[0])[1]
    thr = best["threshold"]

    # TEST — evaluated once, at the validation-chosen threshold
    pt = clf.predict_proba(vec.transform(X(te)))[:, 1]
    test_m = metrics(y[te], pt, thr)

    # Compact saved model: vocab + idf + LR weights (int8-quantized coef check too)
    vocab = vec.get_feature_names_out()
    coef = clf.coef_[0].astype(np.float32)
    model_path = os.path.join(HERE, "model.npz")
    np.savez_compressed(model_path, vocab=np.asarray(vocab, dtype="U64"), idf=vec.idf_.astype(np.float32),
                        coef=coef, intercept=np.float32(clf.intercept_[0]),
                        threshold=np.float32(thr))

    # ONNX export of the full pipeline (best effort)
    onnx_path, onnx_err = os.path.join(HERE, "model.onnx"), None
    try:
        from skl2onnx import to_onnx
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("tfidf", vec), ("lr", clf)])
        onx = to_onnx(pipe, X(tr)[:1], options={"zipmap": False},
                      target_opset={"": 14, "ai.onnx.ml": 2})
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
    except Exception as e:
        onnx_err = str(e); onnx_path = None

    report = {
        "dataset": "UCI SMS Spam Collection", "n_total_after_dedup": int(len(texts)),
        "label_conflicts_dropped": int(conflicts),
        "split_sizes": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "test_spam": int(y[te].sum()), "test_ham": int((y[te] == 0).sum()),
        "validation_at_chosen_threshold": best,
        "test_once": test_m,
        "model_npz_bytes": os.path.getsize(model_path),
        "onnx_bytes": os.path.getsize(onnx_path) if onnx_path else None,
        "onnx_error": onnx_err,
        "vocab_size": int(len(vocab)),
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
