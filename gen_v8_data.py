#!/usr/bin/env python3
"""Generate the v8 datasets (deterministic code only; no LLM text).

Pre-registered in PROVENANCE.md (v8 section) BEFORE this data existed.

Outputs under data/v8/:

1. fam8_train.jsonl / fam8_validation.jsonl / fam8_test.jsonl —
   augmentations of the NEW training families in seeds_v8.json
   (gig_task, crypto_unlock), v4 method: split BY TEMPLATE (index 1
   -> validation only, index 3 -> test only, rest -> train), variant
   counts 90/50/70 per template, augment_v4 slot-filling.

2. recomb_train.jsonl / recomb_validation.jsonl — cross-family seed
   recombinations (gen_v6_data method) over the MERGED seed pool
   (seeds_v4 + seeds_v8). Held out: romance + grandparent contribute
   NO parts. gig_task / crypto_unlock contribute (train-split parts
   to train, validation-split parts to validation). Fresh v8 seeds.

3. obf_eval_scam.jsonl / obf_eval_legit.jsonl — obfuscation set C.
   Base rows are disjoint from BOTH v6's eval and B-v7: both earlier
   selections are reproduced exactly and excluded; set C uses ALL
   remaining seed-42 held-out bases (765 IMC25 test positives, 78 UCI
   test ham). Transforms are the never-trained set-B mechanisms at
   the B-v7 parameter variants (gen_v7_data.OBF_B7), fresh seeds.
   Built BEFORE training; evaluated once.

4. hf_eval.jsonl — held-out-family exam: romance + grandparent,
   70 variants per family generated from ALL their seed templates
   (no template of these families is ever train/validation).
   Neither v8 nor v5 ever trained on these families.

5. manifest.json — counts, families, sources pointer, seeds.
"""
import json
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

import augment_v4
import gen_v6_data
import gen_v7_data
from train_v2 import load_uci, load_modern

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "v8")
TRAIN_FAMILIES = ["gig_task", "crypto_unlock"]
EXAM_FAMILIES = ["romance", "grandparent"]
VARIANTS = {"train": 90, "validation": 50, "test": 70}
EXAM_VARIANTS_PER_FAMILY = 70


def split_of(idx):
    if idx == 1:
        return "validation"
    if idx == 3:
        return "test"
    return "train"


def gen_family_rows(fam, split, per_template):
    rows = []
    for idx, tpl in enumerate(fam["templates"]):
        if split_of(idx) != split:
            continue
        rng = random.Random(f"v8-{split}-{tpl['id']}")
        seen = set()
        for _ in range(per_template * 3):
            txt = augment_v4.augment_template(tpl["text"],
                                              fam["family"], rng)
            if txt in seen:
                continue
            seen.add(txt)
            rows.append({"text": txt, "family": fam["family"],
                         "template_id": tpl["id"]})
            if len(seen) >= per_template:
                break
    return rows


def gen_exam_rows(fam, n_rows):
    """Exam rows from ALL templates of a held-out family."""
    rows, seen = [], set()
    per = n_rows // len(fam["templates"])
    for tpl in fam["templates"]:
        rng = random.Random(f"v8-exam-{tpl['id']}")
        made = 0
        for _ in range(per * 4):
            txt = augment_v4.augment_template(tpl["text"],
                                              fam["family"], rng)
            if txt in seen:
                continue
            seen.add(txt)
            rows.append({"text": txt, "label": 1,
                         "family": fam["family"],
                         "template_id": tpl["id"]})
            made += 1
            if made >= per:
                break
    return rows


def earlier_obf_bases(modern_test, test_ham):
    """Reproduce v6's and B-v7's base selections exactly; return the
    union of used base keys (v6 first-800/300; B-v7 next-800/300)."""
    used_s, used_l = gen_v7_data.v6_eval_bases(modern_test, test_ham)
    scam_pool = [t for t in modern_test
                 if t.strip().lower() not in used_s]
    legit_pool = [t for t in test_ham
                  if t.strip().lower() not in used_l]
    rng = random.Random("v7-obf-eval")
    rng.shuffle(scam_pool)
    rng.shuffle(legit_pool)
    used_s |= {t.strip().lower() for t in scam_pool[:800]}
    used_l |= {t.strip().lower() for t in legit_pool[:300]}
    return used_s, used_l


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds4 = json.load(open(os.path.join(HERE, "seeds_v4.json"),
                            encoding="utf-8"))
    seeds8 = json.load(open(os.path.join(HERE, "seeds_v8.json"),
                            encoding="utf-8"))
    fams8 = {f["family"]: f for f in seeds8["families"]}

    # ---- 1. new training-family rows (v4 split rule) ----
    fam_rows = {}
    for split in ("train", "validation", "test"):
        rows = []
        for fname in TRAIN_FAMILIES:
            rows += gen_family_rows(fams8[fname], split,
                                    VARIANTS[split])
        fam_rows[split] = rows

    # ---- 2. recombinations over the merged pool ----
    merged = {"families": seeds4["families"] + seeds8["families"],
              "meta": {}}
    gen_v6_data.HELD_OUT_FAMILIES = EXAM_FAMILIES
    recomb_train, pool_train = gen_v6_data.build_recombinations(
        merged, "train", 2600, "v8-recomb-train")
    recomb_val, pool_val = gen_v6_data.build_recombinations(
        merged, "validation", 400, "v8-recomb-val")
    existing = set()
    for split in ("train", "validation", "test"):
        path = os.path.join(HERE, "data", "v4", f"fam_{split}.jsonl")
        for line in open(path, encoding="utf-8"):
            existing.add(json.loads(line)["text"].strip().lower())
    for rows in fam_rows.values():
        existing |= {r["text"].strip().lower() for r in rows}
    recomb_train = [r for r in recomb_train
                    if r["text"].strip().lower() not in existing]
    recomb_val = [r for r in recomb_val
                  if r["text"].strip().lower() not in existing]

    # ---- held-out splits (same discipline as v1-v7) ----
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42,
                              stratify=y[tmp])
    keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, _ = load_modern(keys)
    test_ham = [texts[i] for i in te if y[i] == 0]

    # ---- 3. obfuscation set C: all remaining disjoint bases ----
    used_s, used_l = earlier_obf_bases(modern["test"], test_ham)
    scam_pool = [t for t in modern["test"]
                 if t.strip().lower() not in used_s]
    legit_pool = [t for t in test_ham
                  if t.strip().lower() not in used_l]
    rng = random.Random("v8-obf-eval")
    rng.shuffle(scam_pool)
    rng.shuffle(legit_pool)
    names = sorted(gen_v7_data.OBF_B7)
    obf_scam, obf_legit = [], []
    for i, t in enumerate(scam_pool):
        name = names[i % len(names)]
        obf_scam.append({
            "text": gen_v7_data.OBF_B7[name](
                t, random.Random(f"v8-obf-scam-{i}")),
            "label": 1, "transform": name, "base": "imc25_test_rest"})
    for i, t in enumerate(legit_pool):
        name = names[i % len(names)]
        obf_legit.append({
            "text": gen_v7_data.OBF_B7[name](
                t, random.Random(f"v8-obf-legit-{i}")),
            "label": 0, "transform": name, "base": "uci_test_ham_rest"})

    # ---- 4. held-out-family exam: romance + grandparent ----
    hf_rows = []
    for fname in EXAM_FAMILIES:
        hf_rows += gen_exam_rows(fams8[fname],
                                 EXAM_VARIANTS_PER_FAMILY)

    def dump(name, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("fam8_train.jsonl", fam_rows["train"])
    dump("fam8_validation.jsonl", fam_rows["validation"])
    dump("fam8_test.jsonl", fam_rows["test"])
    dump("recomb_train.jsonl", recomb_train)
    dump("recomb_validation.jsonl", recomb_val)
    dump("obf_eval_scam.jsonl", obf_scam)
    dump("obf_eval_legit.jsonl", obf_legit)
    dump("hf_eval.jsonl", hf_rows)

    manifest = {
        "training_families_new": TRAIN_FAMILIES,
        "exam_families_held_out": EXAM_FAMILIES,
        "exam_family_note": "romance + grandparent are NEW in "
                            "seeds_v8.json (they do not exist in "
                            "seeds_v4.json). Excluded from ALL v8 "
                            "training and validation (family rows and "
                            "recombination parts). hf_eval = 70 "
                            "generated variants per family from all "
                            "their templates. Neither v8 nor v5 ever "
                            "trained on these families, so the v5 "
                            "baseline is a true never-seen-family "
                            "score.",
        "fam8_rows": {s: len(r) for s, r in fam_rows.items()},
        "fam8_split_rule": "per family: template index 1 -> "
                           "validation, index 3 -> test, others -> "
                           "train (v4 rule); variants 90/50/70",
        "recomb_train_rows": len(recomb_train),
        "recomb_validation_rows": len(recomb_val),
        "recomb_pools": {"train": pool_train, "validation": pool_val},
        "recomb_note": "merged seed pool (seeds_v4 + seeds_v8); "
                       "romance/grandparent contribute no parts; "
                       "gig_task/crypto_unlock contribute train/val "
                       "parts per the split rule",
        "obf_eval_C": {
            "scam_rows": len(obf_scam), "legit_rows": len(obf_legit),
            "variants": names,
            "base_disjoint_from_v6_and_B_v7": True,
            "base": "ALL remaining IMC25 test positives / UCI test "
                    "ham after reproducing and excluding the v6 and "
                    "B-v7 selections (765 scam / 78 legit remain of "
                    "2365 / 678); fresh seeds v8-obf-*; built before "
                    "v8 training; evaluated once",
        },
        "hf_eval_rows": len(hf_rows),
        "sources": "seeds_v8.json meta.sources (FTC / FBI IC3 / FCC "
                   "/ FinCEN US-government consumer publications)",
        "generation": "deterministic code only (gen_v8_data.py, "
                      "gen_v6_data.py, gen_v7_data.py, augment_v4.py, "
                      "obfuscate_v6.py); no LLM text",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
