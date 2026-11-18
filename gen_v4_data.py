#!/usr/bin/env python3
"""Generate the v4 datasets (deterministic, seeded; no LLM text).

Outputs under data/v4/:
- fam_train.jsonl / fam_val.jsonl / fam_test.jsonl: augmentations of the
  government-sourced seed templates in seeds_v4.json. Split is BY TEMPLATE
  (per family: template index 1 -> validation only, index 3 -> test only,
  the rest -> train), so the held-out 2026-family test contains seed
  templates the model never trains on.
- imc25_train_demasked.jsonl: every IMC25 train-split text with the
  researchers' placeholders replaced by code-generated raw forms
  (augment_v4.demask) — the de-masking fix for v3's known weakness.
- manifest.json: counts and generation parameters.
"""
import json
import os
import random

import augment_v4
from train_v2 import load_uci, load_modern

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "v4")
VARIANTS = {"train": 90, "validation": 50, "test": 70}


def split_of(idx):
    if idx == 1:
        return "validation"
    if idx == 3:
        return "test"
    return "train"


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = json.load(open(os.path.join(HERE, "seeds_v4.json"),
                           encoding="utf-8"))
    counts = {}
    handles = {s: open(os.path.join(OUT, f"fam_{s}.jsonl"), "w",
                       encoding="utf-8")
               for s in ("train", "validation", "test")}
    n_templates = {"train": 0, "validation": 0, "test": 0}
    for fam in seeds["families"]:
        fname = fam["family"]
        for idx, tpl in enumerate(fam["templates"]):
            split = split_of(idx)
            n_templates[split] += 1
            rng = random.Random(f"v4-{split}-{tpl['id']}")
            seen = set()
            for _ in range(VARIANTS[split]):
                txt = augment_v4.augment_template(tpl["text"], fname, rng)
                if txt in seen:
                    continue
                seen.add(txt)
                handles[split].write(json.dumps(
                    {"text": txt, "family": fname,
                     "template_id": tpl["id"]}) + "\n")
                counts[split] = counts.get(split, 0) + 1
    for h in handles.values():
        h.close()

    # De-masked variant of the IMC25 train split (same splits as v2/v3).
    import numpy as np
    from sklearn.model_selection import train_test_split
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42,
                              stratify=y[tmp])
    keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, _ = load_modern(keys)
    n_demasked = 0
    with open(os.path.join(OUT, "imc25_train_demasked.jsonl"), "w",
              encoding="utf-8") as f:
        for i, t in enumerate(modern["train"]):
            rng = random.Random(f"demask-train-{i}")
            f.write(json.dumps({"text": augment_v4.demask(t, rng)}) + "\n")
            n_demasked += 1

    manifest = {
        "family_rows": counts,
        "templates_per_split": n_templates,
        "imc25_train_demasked_rows": n_demasked,
        "variants_per_template": VARIANTS,
        "split_rule": "per family: template index 1 -> validation, "
                      "index 3 -> test, others -> train",
        "rng": "python random.Random seeded per template / per row "
               "(seeds recorded in gen_v4_data.py)",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
