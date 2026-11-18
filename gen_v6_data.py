#!/usr/bin/env python3
"""Generate the v6 datasets (deterministic code only; no LLM text).

Outputs under data/v6/:

1. recomb_train.jsonl / recomb_validation.jsonl — NOVEL scam texts that
   never existed as seeds: openers / bodies / calls-to-action extracted
   from the v4 government-sourced seed templates (sentence-split, slot-
   filled) and recombined ACROSS families. Only the 9 TRAINED families
   contribute parts; train parts come from train-split templates and
   validation parts from validation-split templates (v4 split rule), so
   no test-split template leaks in either.

2. obf_eval_scam.jsonl / obf_eval_legit.jsonl — the NEW obfuscation
   held-out set, built BEFORE training and evaluated once at the end.
   Base texts are held-out rows (IMC25 test positives, UCI test ham)
   transformed with obfuscate_v6 set B (homoglyph / symbol leet /
   letter repeat / vowel drop) — mechanisms never used in training.

3. hf_eval.jsonl — the NEW held-out-FAMILY set: the existing v4
   fam_test rows for the two families v6 excludes from training
   entirely (voter, jury_duty). Neither model trained on these
   templates; v5 saw other templates of these families, v6 sees none.

4. manifest.json — counts, held-out families, transform sets, seeds.
"""
import json
import os
import random
import re

import numpy as np
from sklearn.model_selection import train_test_split

import augment_v4
import obfuscate_v6
from train_v2 import load_uci, load_modern

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "v6")
HELD_OUT_FAMILIES = ["voter", "jury_duty"]
RECOMB_TRAIN_N = 2600
RECOMB_VAL_N = 400
OBF_SCAM_N = 800
OBF_LEGIT_N = 300

CTA_HINTS = ("http", ".com", "bit.ly", "tinyurl", "visit", "click",
             "call", "reply", "text ", "pay at", "go to", "555")
PHONE_RE = re.compile(r"\d{3}[-. ]?\d{3}[-. ]?\d{4}|\(\d{3}\)")


def split_of(idx):
    if idx == 1:
        return "validation"
    if idx == 3:
        return "test"
    return "train"


def is_cta(sentence):
    s = sentence.lower()
    return any(h in s for h in CTA_HINTS) or bool(PHONE_RE.search(s))


def sentence_parts(template_text, family, rng):
    """Fill slots, split into sentences, classify opener/body/cta."""
    filled = augment_v4.augment_template(template_text, family, rng)
    sents = [s.strip() for s in re.split(r"(?<=[.!])\s+", filled)
             if len(s.strip()) >= 8]
    openers, bodies, ctas = [], [], []
    for i, s in enumerate(sents):
        if is_cta(s):
            ctas.append(s)
        elif i == 0:
            openers.append(s)
        else:
            bodies.append(s)
    return openers, bodies, ctas


def build_recombinations(seeds, split, n_rows, rng_tag):
    openers, bodies, ctas = [], [], []
    for fam in seeds["families"]:
        fname = fam["family"]
        if fname in HELD_OUT_FAMILIES:
            continue
        for idx, tpl in enumerate(fam["templates"]):
            if split_of(idx) != split:
                continue
            rng = random.Random(f"v6-parts-{split}-{tpl['id']}")
            for _ in range(12):
                o, b, c = sentence_parts(tpl["text"], fname, rng)
                openers += o
                bodies += b
                ctas += c
    openers = sorted(set(openers))
    bodies = sorted(set(bodies))
    ctas = sorted(set(ctas))
    rng = random.Random(rng_tag)
    rows, seen = [], set()
    tries = 0
    while len(rows) < n_rows and tries < n_rows * 30:
        tries += 1
        parts = [rng.choice(openers)]
        if bodies and rng.random() < 0.85:
            parts.append(rng.choice(bodies))
        parts.append(rng.choice(ctas))
        txt = " ".join(parts)
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": txt, "label": 1, "source": "recombination"})
    return rows, {"openers": len(openers), "bodies": len(bodies),
                  "ctas": len(ctas), "rows": len(rows)}


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = json.load(open(os.path.join(HERE, "seeds_v4.json"),
                           encoding="utf-8"))

    recomb_train, pool_train = build_recombinations(
        seeds, "train", RECOMB_TRAIN_N, "v6-recomb-train")
    recomb_val, pool_val = build_recombinations(
        seeds, "validation", RECOMB_VAL_N, "v6-recomb-val")

    # Drop any recombination that exactly equals an existing family row.
    existing = set()
    for split in ("train", "validation", "test"):
        path = os.path.join(HERE, "data", "v4", f"fam_{split}.jsonl")
        for line in open(path, encoding="utf-8"):
            existing.add(json.loads(line)["text"].strip().lower())
    recomb_train = [r for r in recomb_train
                    if r["text"].strip().lower() not in existing]
    recomb_val = [r for r in recomb_val
                  if r["text"].strip().lower() not in existing]

    # ---- held-out base texts (same splits as v1-v5, seed 42) ----
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42,
                              stratify=y[tmp])
    keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, _ = load_modern(keys)
    test_ham = [texts[i] for i in te if y[i] == 0]

    # ---- obfuscation eval (set B transforms, one per row, cycling) ----
    rng = random.Random("v6-obf-eval")
    scam_base = modern["test"][:]
    rng.shuffle(scam_base)
    scam_base = scam_base[:OBF_SCAM_N]
    legit_base = test_ham[:]
    rng.shuffle(legit_base)
    legit_base = legit_base[:OBF_LEGIT_N]
    names = sorted(obfuscate_v6.EVAL_TRANSFORMS)
    obf_scam, obf_legit = [], []
    for i, t in enumerate(scam_base):
        name = names[i % len(names)]
        obf_scam.append({
            "text": obfuscate_v6.EVAL_TRANSFORMS[name](
                t, random.Random(f"v6-obf-scam-{i}")),
            "label": 1, "transform": name, "base": "imc25_test"})
    for i, t in enumerate(legit_base):
        name = names[i % len(names)]
        obf_legit.append({
            "text": obfuscate_v6.EVAL_TRANSFORMS[name](
                t, random.Random(f"v6-obf-legit-{i}")),
            "label": 0, "transform": name, "base": "uci_test_ham"})

    # ---- held-out-family eval: fam_test rows of the held-out families ----
    hf_rows = []
    path = os.path.join(HERE, "data", "v4", "fam_test.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["family"] in HELD_OUT_FAMILIES:
            hf_rows.append({"text": r["text"], "label": 1,
                            "family": r["family"],
                            "template_id": r["template_id"]})

    def dump(name, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("recomb_train.jsonl", recomb_train)
    dump("recomb_validation.jsonl", recomb_val)
    dump("obf_eval_scam.jsonl", obf_scam)
    dump("obf_eval_legit.jsonl", obf_legit)
    dump("hf_eval.jsonl", hf_rows)

    manifest = {
        "held_out_families": HELD_OUT_FAMILIES,
        "held_out_family_note": "voter + jury_duty excluded from ALL v6 "
                                "training and validation data (family "
                                "rows, recombination parts). hf_eval = "
                                "their v4 fam_test rows (templates "
                                "neither v5 nor v6 trained on).",
        "recomb_train_rows": len(recomb_train),
        "recomb_validation_rows": len(recomb_val),
        "recomb_pools": {"train": pool_train, "validation": pool_val},
        "obf_eval": {
            "scam_rows": len(obf_scam), "legit_rows": len(obf_legit),
            "transforms_set_B_eval_only": names,
            "transforms_set_A_training_only": [
                "leet_digits", "space_split", "punct_insert", "case_flip",
                "zero_width", "txt_speak (legit only)"],
            "base": "IMC25 test positives / UCI test ham, seed-42 splits; "
                    "built before v6 training; evaluated once",
        },
        "hf_eval_rows": len(hf_rows),
        "generation": "deterministic code only (gen_v6_data.py, "
                      "obfuscate_v6.py, augment_v4.py); no LLM text",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
