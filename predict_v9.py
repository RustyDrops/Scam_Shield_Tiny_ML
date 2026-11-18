#!/usr/bin/env python3
"""Runtime inference for Scam Shield v9 from model_v9.npz only.
Reimplements sklearn TF-IDF for both branches of the v3 FeatureUnion:
word 1-2 grams and char_wb 3-5 grams, each sublinear-tf, L2-normalized
separately (as TfidfVectorizer does), concatenated for the LR head."""
import os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_m = np.load(os.path.join(HERE, "model_v9.npz"), allow_pickle=False)
WV = {t: i for i, t in enumerate(_m["word_vocab"].tolist())}
CV = {t: i for i, t in enumerate(_m["char_vocab"].tolist())}
WIDF, CIDF, COEF = _m["word_idf"], _m["char_idf"], _m["coef"]
INTERCEPT, THR = float(_m["intercept"]), float(_m["threshold"])
TOK = re.compile(r"(?u)\b\w[\w']+\b")
WS = re.compile(r"\s+")

def _tfidf(counts, idf, size):
    x = np.zeros(size, dtype=np.float32)
    for i, c in counts.items():
        x[i] = c
    nz = x > 0
    x[nz] = (1.0 + np.log(x[nz])) * idf[nz]
    n = np.linalg.norm(x)
    if n > 0:
        x /= n
    return x

def prob_scam(text: str) -> float:
    low = text.lower()
    toks = TOK.findall(low)
    wcounts = {}
    for f in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]:
        i = WV.get(f)
        if i is not None:
            wcounts[i] = wcounts.get(i, 0) + 1
    ccounts = {}
    for w in WS.sub(" ", low).split():
        w = " " + w + " "
        for n in (3, 4, 5):
            for off in range(0, len(w) - n + 1):
                i = CV.get(w[off:off + n])
                if i is not None:
                    ccounts[i] = ccounts.get(i, 0) + 1
    x = np.concatenate([_tfidf(wcounts, WIDF, len(WV)), _tfidf(ccounts, CIDF, len(CV))])
    z = float(x @ COEF + INTERCEPT)
    return 1.0 / (1.0 + np.exp(-z))

def classify(text: str):
    p = prob_scam(text)
    return ("scam/spam" if p >= THR else "legitimate"), round(p, 4), THR

if __name__ == "__main__":
    print(classify(" ".join(sys.argv[1:])))
