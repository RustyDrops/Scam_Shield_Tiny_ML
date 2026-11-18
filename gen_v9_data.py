#!/usr/bin/env python3
"""Generate the v9 datasets (deterministic code only; no LLM text).

Pre-registered in PROVENANCE.md (v9 section) BEFORE this data existed.

v9 = v5's exact recipe and training data + ADDED data only:

Outputs under data/v9/:

1. fam9_train.jsonl / fam9_validation.jsonl / fam9_test.jsonl —
   conversational scam families, augment_v4 slot-filling, split BY
   TEMPLATE (index 1 -> validation, index 3 -> internal test, rest
   -> train), variant counts 90/50/70 per template:
   - romance + grandparent (seeds_v8.json — v8's spent exam
     families, now eligible for training)
   - wrong_number_opener (seeds_v9.json — NEW: wrong-number
     conversational openers pivoting to a crypto/mentor pitch)

2. legit_conv_train.jsonl / legit_conv_validation.jsonl — NEW
   hard-legit conversational data (this file's templates, v7 style):
   genuine wrong-number replies, wrong-person apologies, family and
   friendly chat. Split BY TEMPLATE within each category
   (index % 4 == 1 -> validation), 50 variants per train template,
   30 per validation template, exact-match guarded against all
   frozen validation/test keys and all v9 scam rows.

3. obf_eval_scam.jsonl / obf_eval_legit.jsonl — obfuscation set D.
   The IMC25-test / UCI-test base pools are exhausted by sets
   v6/B-v7/C, so set D draws NEW bases from the VALIDATION pools
   (800 IMC25 validation positives + 300 UCI validation ham, seeded
   sample), transformed with the never-trained set-B mechanisms at
   the B-v7 variants (gen_v7_data.OBF_B7), fresh v9 seeds. Built
   BEFORE training; evaluated once.

4. hf_eval.jsonl — held-out-family exam: charity_relief +
   package_delivery (seeds_v9.json), 70 variants per family from
   ALL their templates. Excluded from ALL v9 training/validation.
   Neither v9 nor v5 trained on these seed families (disclosed in
   PROVENANCE: delivery register is partially seen via v5
   transactional templates and IMC25's delivery category).

5. manifest.json — counts, families, sources pointer, seeds.
"""
import json
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

import augment_v4
import gen_v7_data
from train_v2 import load_uci, load_modern

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "v9")
TRAIN_FAMILIES_V8 = ["romance", "grandparent"]
TRAIN_FAMILIES_V9 = ["wrong_number_opener"]
EXAM_FAMILIES = ["charity_relief", "package_delivery"]
VARIANTS = {"train": 90, "validation": 50, "test": 70}
EXAM_VARIANTS_PER_FAMILY = 70
OBF_SCAM_N = 800
OBF_LEGIT_N = 300

# URL cores for the new families (augment_v4 stays untouched)
augment_v4.URL_CORES.setdefault("wrong_number_opener",
                                ["chatmeet", "friendlink", "newchat"])
augment_v4.URL_CORES.setdefault("charity_relief",
                                ["reliefaid", "donaterelief", "charityhelp",
                                 "giveaid", "veteransaid", "relieffund"])
augment_v4.URL_CORES.setdefault("package_delivery",
                                ["usps", "fedex", "ups", "dhl", "trackpkg",
                                 "redelivery", "parcelhold", "deliverypost"])


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
        rng = random.Random(f"v9-{split}-{tpl['id']}")
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
        rng = random.Random(f"v9-exam-{tpl['id']}")
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


# ---------------- hard-legit conversational templates ----------------
# Slot-filled code only (v7 style). INFORMAL ONLY: no links, no asks
# for money/codes/secrets. A few ordinary family money mentions are
# deliberate (family talking about money is not a scam tell).

def _amt(rng):
    return f"${rng.randint(5, 200)}"


LEGIT_CONV_TEMPLATES = [
    ("wrong_number_reply", [
        "Sorry, you have the wrong number.",
        "Wrong number, sorry! There is no {name} here.",
        "I think you texted the wrong person. Hope you find {name}!",
        "This is not {name}'s phone anymore, sorry.",
        "You have got the wrong number, this is {name} speaking.",
        "Nope, wrong number. Good luck though!",
        "Sorry, who is this? I do not recognize this number.",
        "You might have the number wrong, I do not know a {name}.",
        "Wrong chat, sorry! I think that was meant for {name2}.",
        "Hi, I think this message was meant for {name}.",
        "I get texts for {name} all the time, this is {name2}, not her.",
    ]),
    ("wrong_person_apology", [
        "Oops, sorry, that last text was meant for my sister. Wrong person!",
        "My bad, wrong chat! Please ignore what I just sent, that was for {name2}.",
        "Sorry everyone, pocket text. This is {name2}, wrong group!",
        "Ignore my last message, I sent it to the wrong person. Sorry {name}!",
        "That photo was meant for my mom, sorry! Wrong thread.",
        "Wrong person, my apologies, I was texting my coworker {name2}.",
        "Sorry, I mixed up my contacts. That message was not for you.",
        "Oops! Wrong {name}. Sorry about that.",
        "Please disregard my last text, sent to the wrong number by mistake.",
        "My apologies, I have two contacts named {name} and I picked the wrong one.",
        "Sorry for the confusion, that message was for {name2}, not you.",
    ]),
    ("family_chat", [
        "Hi Mom, dinner on {day} is at our house this week. Can you bring your {food}?",
        "Grandma, I sent you {amt} for your birthday through the bank app. Love you!",
        "Dad, the grandkids made you a card. I will drop it by on {day}.",
        "Can you pick up {name} from school at {time}? I am stuck at work.",
        "We prayed for you at church this morning. Hope the doctor visit goes well on {day}.",
        "{name} lost a tooth today! The tooth fairy is coming tonight.",
        "Call me when you get this, just want to hear your voice. Love, {name2}",
        "The baby is finally sleeping through the night. {name2} will send pictures tomorrow.",
        "Do not forget Grandpa's birthday dinner on {day} at {time}.",
        "I mailed your package of cookies today, should arrive by {day}.",
        "Tell {name} we said happy birthday! A card is in the mail.",
    ]),
    ("friendly_chat", [
        "Hey {name}, your dog got out again, he is in our yard. No rush.",
        "Book club is at {place} on {day} evening, bring a snack if you can.",
        "Thanks for watering our plants while we were away, you are the best!",
        "Can I borrow your ladder this weekend? I will return it on {day}.",
        "The HOA meeting moved to {time} on {day}, same clubhouse.",
        "Game night on {day}? We have that new card game to try.",
        "Hi {name}, your car lights are on in the driveway, just a heads up!",
        "Hi {name}, I have extra tomatoes from the garden if you want some.",
        "Your ride to the airport on {day} is all set, I will be there at {time}.",
        "Congrats on the new job, {name}! Dinner is on us next week.",
        "Walking the trail at {place} at {time} if you want to join us.",
    ]),
]


_NAMES = sorted(set(gen_v7_data.FIRST) | set(augment_v4.FIRST_NAMES))
_PLACES = ["the park", "the library", "the coffee shop", "church",
           "the farmers market", "the community center", "the lake",
           "the school gym", "our place", "the diner on Main"]
_FOODS = ["potato salad", "a casserole", "banana bread", "a fruit pie",
          "coleslaw", "baked beans", "a green salad", "cornbread",
          "deviled eggs", "lemonade"]


def fill_conv(tpl, rng):
    return tpl.format(name=rng.choice(_NAMES),
                      name2=rng.choice(_NAMES),
                      day=rng.choice(gen_v7_data.DAYS),
                      time=rng.choice(gen_v7_data.TIMES),
                      place=rng.choice(_PLACES),
                      food=rng.choice(_FOODS),
                      amt=_amt(rng))


def build_legit_conv(guard_keys):
    train, val = [], []
    for cat, tpls in LEGIT_CONV_TEMPLATES:
        for j, tpl in enumerate(tpls):
            split = "validation" if j % 4 == 1 else "train"
            n_var = 30 if split == "validation" else 50
            rng = random.Random(f"v9-legit-{cat}-{j}")
            seen = set()
            for _ in range(n_var * 3):
                t = fill_conv(tpl, rng)
                k = t.strip().lower()
                if k in seen or k in guard_keys:
                    continue
                seen.add(k)
                row = {"text": t, "label": 0, "category": cat,
                       "template_id": f"{cat}-{j}",
                       "source": "legit_conv_v9"}
                (val if split == "validation" else train).append(row)
                if len(seen) >= n_var:
                    break
    return train, val


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds8 = json.load(open(os.path.join(HERE, "seeds_v8.json"),
                            encoding="utf-8"))
    seeds9 = json.load(open(os.path.join(HERE, "seeds_v9.json"),
                            encoding="utf-8"))
    fams = {f["family"]: f
            for f in seeds8["families"] + seeds9["families"]}

    # ---- 1. conversational scam family rows (v4 split rule) ----
    fam_rows = {}
    train_fams = TRAIN_FAMILIES_V8 + TRAIN_FAMILIES_V9
    for split in ("train", "validation", "test"):
        rows = []
        for fname in train_fams:
            rows += gen_family_rows(fams[fname], split,
                                    VARIANTS[split])
        fam_rows[split] = rows

    # ---- 2. held-out-family exam: charity_relief + package ----
    hf_rows = []
    for fname in EXAM_FAMILIES:
        hf_rows += gen_exam_rows(fams[fname],
                                 EXAM_VARIANTS_PER_FAMILY)

    # ---- held-out splits (same discipline as v1-v8) ----
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42,
                              stratify=y[tmp])
    keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, _ = load_modern(keys)
    val_ham = [texts[i] for i in va if y[i] == 0]

    # ---- 3. hard-legit conversational (guarded) ----
    guard = set(keys) | {t.strip().lower() for t in modern["test"]}
    for split in ("train", "validation", "test"):
        path = os.path.join(HERE, "data", "v4", f"fam_{split}.jsonl")
        for line in open(path, encoding="utf-8"):
            guard.add(json.loads(line)["text"].strip().lower())
    for split in ("train", "validation", "test"):
        path = os.path.join(HERE, "data", "v5", f"txn_{split}.jsonl")
        for line in open(path, encoding="utf-8"):
            guard.add(json.loads(line)["text"].strip().lower())
    for rows in list(fam_rows.values()) + [hf_rows]:
        guard |= {r["text"].strip().lower() for r in rows}
    legit_train, legit_val = build_legit_conv(guard)

    # ---- 4. obfuscation set D: bases from the VALIDATION pools ----
    # A transformed row is EXCLUDED if it is a no-op (transform left
    # the text unchanged — not an obfuscation test at all) or if it
    # exactly matches ANY clean train/validation text; the walk
    # continues down the shuffled pool until the counts are met.
    forbidden = set(guard)
    forbidden |= {texts[i].strip().lower() for i in list(tr)}
    forbidden |= {t.strip().lower() for t in modern["train"]}
    forbidden |= {t.strip().lower() for t in modern["validation"]}
    for split in ("train", "validation"):
        path = os.path.join(HERE, "data", "v4",
                            f"imc25_train_demasked.jsonl"
                            if split == "train" else f"fam_{split}.jsonl")
        for line in open(path, encoding="utf-8"):
            forbidden.add(json.loads(line)["text"].strip().lower())
    forbidden |= {r["text"].strip().lower() for r in legit_train}
    forbidden |= {r["text"].strip().lower() for r in legit_val}
    rng = random.Random("v9-obf-eval")
    scam_pool = modern["validation"][:]
    rng.shuffle(scam_pool)
    legit_pool = val_ham[:]
    rng.shuffle(legit_pool)
    names = sorted(gen_v7_data.OBF_B7)

    def pick_obf(pool, n_want, label, base, tag):
        rows, seen = [], set()
        i = 0
        for t in pool:
            if len(rows) >= n_want:
                break
            name = names[i % len(names)]
            txt = gen_v7_data.OBF_B7[name](
                t, random.Random(f"v9-obf-{tag}-{i}"))
            i += 1
            k = txt.strip().lower()
            if k == t.strip().lower() or k in forbidden or k in seen:
                continue
            seen.add(k)
            rows.append({"text": txt, "label": label,
                         "transform": name, "base": base})
        assert len(rows) >= n_want, f"set D pool short: {len(rows)}"
        return rows

    obf_scam = pick_obf(scam_pool, OBF_SCAM_N, 1,
                        "imc25_validation", "scam")
    obf_legit = pick_obf(legit_pool, OBF_LEGIT_N, 0,
                         "uci_validation_ham", "legit")

    def dump(name, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("fam9_train.jsonl", fam_rows["train"])
    dump("fam9_validation.jsonl", fam_rows["validation"])
    dump("fam9_test.jsonl", fam_rows["test"])
    dump("legit_conv_train.jsonl", legit_train)
    dump("legit_conv_validation.jsonl", legit_val)
    dump("obf_eval_scam.jsonl", obf_scam)
    dump("obf_eval_legit.jsonl", obf_legit)
    dump("hf_eval.jsonl", hf_rows)

    per_family = {}
    for split in ("train", "validation", "test"):
        for r in fam_rows[split]:
            per_family.setdefault(r["family"],
                                  {"train": 0, "validation": 0,
                                   "test": 0})
            per_family[r["family"]][split] += 1
    manifest = {
        "training_families_new": train_fams,
        "exam_families_held_out": EXAM_FAMILIES,
        "exam_family_note": "charity_relief appears nowhere in any "
                            "prior round. package_delivery is new as "
                            "a seed-template family; delivery register "
                            "partially seen via v5 transactional "
                            "templates + IMC25 delivery category "
                            "(disclosed in PROVENANCE v9). hf_eval = "
                            "70 generated variants per family from "
                            "ALL their templates; excluded from ALL "
                            "v9 training and validation.",
        "fam9_rows": {s: len(r) for s, r in fam_rows.items()},
        "fam9_per_family": per_family,
        "fam9_split_rule": "per family: template index 1 -> "
                           "validation, index 3 -> internal test, "
                           "others -> train (v4 rule); variants "
                           "90/50/70; seeds v9-{split}-{template_id}",
        "legit_conv": {
            "train_rows": len(legit_train),
            "validation_rows": len(legit_val),
            "categories": [c for c, _ in LEGIT_CONV_TEMPLATES],
            "note": "deterministic slot-filled informal legit "
                    "templates (gen_v9_data.py); split by template "
                    "(index % 4 == 1 -> validation); exact-match "
                    "guarded against all frozen val/test keys, all "
                    "v9 scam rows and the exam rows",
        },
        "obf_eval_D": {
            "scam_rows": len(obf_scam), "legit_rows": len(obf_legit),
            "variants": names,
            "base": "IMC25 VALIDATION positives / UCI VALIDATION ham "
                    "(the test pools were exhausted by sets v6/B-v7/"
                    "C: 800+800+765 of 2365 test positives, 300+300+"
                    "78 of 678 test ham); seeded sample, shuffle seed "
                    "v9-obf-eval, per-row seeds v9-obf-*; transforms "
                    "= gen_v7_data.OBF_B7 (never trained); no-op "
                    "transforms and rows exactly matching any clean "
                    "train/validation text are excluded during the "
                    "pool walk; built before v9 training; evaluated "
                    "once. Clean base "
                    "forms sat in validation sets (threshold "
                    "selection only) for both lineages — disclosed "
                    "in PROVENANCE v9.",
        },
        "hf_eval_rows": len(hf_rows),
        "sources": "seeds_v8.json + seeds_v9.json meta.sources (FTC "
                   "/ FBI IC3 / FCC / FinCEN / USPIS US-government "
                   "publications)",
        "generation": "deterministic code only (gen_v9_data.py, "
                      "augment_v4.py, gen_v7_data.py, obfuscate_v6.py"
                      "); no LLM text",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
