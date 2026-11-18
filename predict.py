#!/usr/bin/env python3
"""Runtime inference for Scam Shield baseline from model.npz only.
Reimplements sklearn TF-IDF (word 1-2 grams, sublinear tf, L2 norm)
so the saved model is self-contained and tiny."""
import os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_m = np.load(os.path.join(HERE, "model.npz"), allow_pickle=False)
VOCAB = {t: i for i, t in enumerate(_m["vocab"].tolist())}
IDF, COEF = _m["idf"], _m["coef"]
INTERCEPT, THR = float(_m["intercept"]), float(_m["threshold"])
TOK = re.compile(r"(?u)\b\w[\w']+\b")

def prob_scam(text: str) -> float:
    toks = TOK.findall(text.lower())
    feats = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    x = np.zeros(len(VOCAB), dtype=np.float32)
    for f in feats:
        i = VOCAB.get(f)
        if i is not None:
            x[i] += 1.0
    nz = x > 0
    x[nz] = (1.0 + np.log(x[nz])) * IDF[nz]  # sublinear tf * idf
    n = np.linalg.norm(x)
    if n > 0:
        x /= n
    z = float(x @ COEF + INTERCEPT)
    return 1.0 / (1.0 + np.exp(-z))

def classify(text: str):
    p = prob_scam(text)
    return ("scam/spam" if p >= THR else "legitimate"), round(p, 4), THR

if __name__ == "__main__":
    print(classify(" ".join(sys.argv[1:])))
