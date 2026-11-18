#!/usr/bin/env python3
"""Generate the v7 datasets (deterministic code only; no LLM text).

Outputs under data/v7/:

1. recomb_train.jsonl / recomb_validation.jsonl — novel cross-family
   seed recombinations (same method as gen_v6_data), but the held-out
   families are now toll + task_job: their templates contribute NO
   parts. voter/jury_duty (v6's spent exam) contribute again. Fresh
   sampling seeds (v7-*).

2. obf_eval_scam.jsonl / obf_eval_legit.jsonl — obfuscation set B-v7.
   Base rows are DISJOINT from v6's eval set: v6's selection (seed
   "v6-obf-eval", first 800 shuffled IMC25 test positives / first 300
   shuffled UCI test ham) is reproduced exactly and those rows are
   excluded; v7 draws the next rows with fresh seeds. Transforms are
   the same never-trained set-B mechanisms at FRESH parameter variants
   (higher word-hit rates, more words touched) plus two stacked pairs
   that v6's eval never used. Built BEFORE training; evaluated once.

3. hf_eval.jsonl — held-out-family exam: v4 fam_test rows for toll +
   task_job (templates neither v5 nor v7 trained on; v7 never sees
   these families at all).

4. legit_train.jsonl / legit_validation.jsonl — NEW hard legit data,
   v5 transactional style, deterministic slot-filled templates:
   security-code informs, appointment/delivery/bank/utility/school
   informs, and short casual texts. INFORM ONLY — no links, no asks
   for secrets or payment. Split BY TEMPLATE (index % 4 == 1 ->
   validation, else train) so validation legit is template-disjoint.

5. manifest.json — counts, families, transform variants, seeds.
"""
import json
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

import gen_v6_data
import obfuscate_v6
from train_v2 import load_uci, load_modern

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "v7")
HELD_OUT_FAMILIES = ["toll", "task_job"]
OBF_SCAM_N = 800
OBF_LEGIT_N = 300


# ---------------- obfuscation set B-v7 variants ----------------
# Same set-B mechanisms as obfuscate_v6 (never trained), fresh params.

def hg_hi(text, rng):
    return obfuscate_v6.homoglyph(text, rng, p_word=0.7)


def sl_hi(text, rng):
    return obfuscate_v6.symbol_leet(text, rng, p_word=0.75)


def lr_hi(text, rng):
    return obfuscate_v6.letter_repeat(text, rng, max_words=6)


def vd_hi(text, rng):
    return obfuscate_v6.vowel_drop(text, rng, max_words=6)


def hg_vd(text, rng):
    return obfuscate_v6.vowel_drop(obfuscate_v6.homoglyph(text, rng),
                                   rng, max_words=5)


def sl_lr(text, rng):
    return obfuscate_v6.letter_repeat(obfuscate_v6.symbol_leet(text, rng),
                                      rng, max_words=5)


OBF_B7 = {"homoglyph_hi": hg_hi, "symbol_leet_hi": sl_hi,
          "letter_repeat_hi": lr_hi, "vowel_drop_hi": vd_hi,
          "homoglyph+vowel_drop": hg_vd, "symbol_leet+letter_repeat": sl_lr}


def v6_eval_bases(modern_test, test_ham):
    """Reproduce gen_v6_data's eval base selection exactly."""
    rng = random.Random("v6-obf-eval")
    scam = modern_test[:]
    rng.shuffle(scam)
    legit = test_ham[:]
    rng.shuffle(legit)
    return ({t.strip().lower() for t in scam[:800]},
            {t.strip().lower() for t in legit[:300]})


# ---------------- hard legit templates (inform-only) ----------------

FIRST = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Mason", "Sophia",
         "Lucas", "Mia", "Ethan", "Grace", "Owen", "Ruby", "Caleb",
         "Nora", "Eli", "Hazel", "Jonah", "Ivy", "Silas"]
SERVICES = ["Google", "Microsoft", "Apple", "Chase", "Venmo", "PayPal",
            "Amazon", "Netflix", "Steam", "WhatsApp", "Uber", "DoorDash",
            "AT&T", "Verizon", "Wells Fargo", "Capital One", "Cash App",
            "Instagram", "Facebook", "your credit union"]
CLINICS = ["Bright Smiles Dental", "Ada Family Clinic", "Vision Center",
           "Main Street Dental", "Lakeside Pediatrics", "City Eye Care",
           "Ada Veterinary Clinic", "QuickCare Clinic"]
COURIERS = ["USPS", "FedEx", "UPS", "Amazon Logistics", "DHL"]
BANKS = ["Chase", "Wells Fargo", "Bank of America", "US Bank",
         "Capital One", "your credit union"]
STORES = ["Walmart", "Target", "Shell", "Dollar General", "CVS",
          "Walgreens", "Home Depot", "Kroger"]
SCHOOLS = ["Ada Public Schools", "Lincoln Elementary",
           "Washington Middle School", "Ada High School"]
UTILITIES = ["OG&E", "City of Ada Utilities", "AT&T", "Verizon",
             "CenterPoint Energy", "Xfinity"]
TIMES = ["9:00 AM", "10:30 AM", "1:15 PM", "2:45 PM", "4:00 PM",
         "5:30 PM", "11:00 AM", "3:20 PM"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "tomorrow", "this Saturday"]


def _code(rng, n=6):
    return "".join(rng.choice("0123456789") for _ in range(n))


def _amt(rng):
    return f"${rng.randint(3, 480)}.{rng.randint(0, 99):02d}"


def _last4(rng):
    return "".join(rng.choice("0123456789") for _ in range(4))


LEGIT_TEMPLATES = [
    # security / 2FA informs
    ("security_code", [
        "{svc}: Your verification code is {code}. It expires in 10 minutes. We will never ask you to share this code.",
        "Your {svc} code is {code}. Do not share it with anyone, including people claiming to work for {svc}.",
        "{svc} security alert: code {code} was requested for your account. If this wasn't you, change your password in the app.",
        "Use {code} to finish signing in to {svc}. This code expires soon and can only be used once.",
        "{svc}: {code} is your one-time passcode. Anyone asking you to read it out or send it on is trying to scam you.",
    ]),
    # appointment informs
    ("appointment", [
        "Reminder: {name} has an appointment at {clinic} on {day} at {time}. Reply C to confirm or call us to reschedule.",
        "{clinic}: This is a reminder of your appointment on {day} at {time}. Please arrive 10 minutes early.",
        "Hi {name}, your appointment with {clinic} is confirmed for {day} at {time}. See you then!",
        "Reminder from {clinic}: {name}'s checkup is scheduled for {day} at {time}. Bring your insurance card.",
        "{clinic} reminder: your visit is {day} at {time}. If you need to cancel, please call the office.",
    ]),
    # delivery informs
    ("delivery", [
        "{courier}: Your package was delivered at {time} today. Tracking {trk}.",
        "Good news — your {courier} package is out for delivery and should arrive by {time} today.",
        "{courier} update: your parcel arrived at the local facility on {day}. No action is needed.",
        "Your package from {store} has shipped via {courier}. Tracking number {trk}.",
        "{courier}: We delivered your package to the front porch at {time}. Thanks for shopping with us.",
    ]),
    # bank informs
    ("bank_alert", [
        "{bank} alert: a purchase of {amt} was made at {store} with your card ending in {last4}. No action needed if you recognize it.",
        "Your {bank} balance is {amt} as of this morning. This is an automated alert you requested.",
        "{bank}: Your statement for this month is ready to view in the app. No payment is due until the 15th.",
        "Alert from {bank}: your direct deposit of {amt} has posted to your account ending in {last4}.",
        "{bank} fraud check: was this you? A charge of {amt} at {store}. Reply YES or NO. We will never ask for your PIN.",
    ]),
    # school informs
    ("school", [
        "{school}: School will dismiss early on {day} at 1:00 PM due to weather. Buses will run their normal routes.",
        "Reminder from {school}: picture day is {day}. Order forms were sent home with students today.",
        "{school}: There is no school on {day} for a teacher work day. Classes resume the following day.",
        "{school} alert: the {day} football game has been moved to 7:30 PM. See you there!",
    ]),
    # utility informs
    ("utility", [
        "{util}: Your monthly bill of {amt} is ready. It will be paid by auto-pay on the 12th as scheduled.",
        "Outage update from {util}: crews are working in your area and power should be restored by {time}.",
        "{util}: We will read your meter on {day}. Please make sure the gate is unlocked. No payment is collected at the door.",
        "{util} reminder: your scheduled maintenance window is {day} between 9 AM and noon. Brief interruptions are possible.",
    ]),
    # casual short texts
    ("casual", [
        "Hey {name}, are we still on for {day}? Let me know!",
        "Running about 10 minutes late, sorry! See you at {time}.",
        "Happy birthday {name}! Hope you have a great day.",
        "Did you pick up milk on the way home?",
        "Practice is moved to {time} on {day}, same field.",
        "Thanks for dinner last night, it was lovely.",
        "Can you grab {name} from school at {time}?",
        "Just landed, taxiing now. See you soon!",
        "The meeting moved to {day} at {time}, same room.",
        "Love you, call when you get a chance.",
        "Don't forget your umbrella, it's pouring out there.",
        "Groceries are in the car, can you help bring them in?",
        "Game night at our place {day}, bring a snack to share!",
        "Your prescription is ready for pickup at the pharmacy.",
        "Church potluck is {day} after the service, sign-up sheet is in the foyer.",
    ]),
]


def fill(tpl, rng):
    return tpl.format(
        svc=rng.choice(SERVICES), clinic=rng.choice(CLINICS),
        courier=rng.choice(COURIERS), bank=rng.choice(BANKS),
        store=rng.choice(STORES), school=rng.choice(SCHOOLS),
        util=rng.choice(UTILITIES), name=rng.choice(FIRST),
        time=rng.choice(TIMES), day=rng.choice(DAYS),
        code=_code(rng), amt=_amt(rng), last4=_last4(rng),
        trk="".join(rng.choice("0123456789") for _ in range(12)))


def build_legit(guard_keys):
    train, val = [], []
    for cat, tpls in LEGIT_TEMPLATES:
        for j, tpl in enumerate(tpls):
            split = "validation" if j % 4 == 1 else "train"
            n_var = 30 if split == "validation" else 50
            rng = random.Random(f"v7-legit-{cat}-{j}")
            seen = set()
            for _ in range(n_var * 3):
                t = fill(tpl, rng)
                k = t.strip().lower()
                if k in seen or k in guard_keys:
                    continue
                seen.add(k)
                row = {"text": t, "label": 0, "category": cat,
                       "template_id": f"{cat}-{j}", "source": "legit_v7"}
                (val if split == "validation" else train).append(row)
                if len(seen) >= n_var:
                    break
    return train, val


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = json.load(open(os.path.join(HERE, "seeds_v4.json"),
                           encoding="utf-8"))

    # ---- recombinations: toll + task_job held out ----
    gen_v6_data.HELD_OUT_FAMILIES = HELD_OUT_FAMILIES
    recomb_train, pool_train = gen_v6_data.build_recombinations(
        seeds, "train", 2600, "v7-recomb-train")
    recomb_val, pool_val = gen_v6_data.build_recombinations(
        seeds, "validation", 400, "v7-recomb-val")
    existing = set()
    for split in ("train", "validation", "test"):
        path = os.path.join(HERE, "data", "v4", f"fam_{split}.jsonl")
        for line in open(path, encoding="utf-8"):
            existing.add(json.loads(line)["text"].strip().lower())
    recomb_train = [r for r in recomb_train
                    if r["text"].strip().lower() not in existing]
    recomb_val = [r for r in recomb_val
                  if r["text"].strip().lower() not in existing]

    # ---- held-out splits (same discipline as v1-v6) ----
    texts, y = load_uci()
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42,
                              stratify=y[tmp])
    keys = {texts[i].strip().lower() for i in list(va) + list(te)}
    modern, _ = load_modern(keys)
    test_ham = [texts[i] for i in te if y[i] == 0]

    # ---- obfuscation set B-v7: bases disjoint from v6's eval ----
    used_scam, used_legit = v6_eval_bases(modern["test"], test_ham)
    rng = random.Random("v7-obf-eval")
    scam_pool = [t for t in modern["test"]
                 if t.strip().lower() not in used_scam]
    legit_pool = [t for t in test_ham
                  if t.strip().lower() not in used_legit]
    rng.shuffle(scam_pool)
    rng.shuffle(legit_pool)
    names = sorted(OBF_B7)
    obf_scam, obf_legit = [], []
    for i, t in enumerate(scam_pool[:OBF_SCAM_N]):
        name = names[i % len(names)]
        obf_scam.append({
            "text": OBF_B7[name](t, random.Random(f"v7-obf-scam-{i}")),
            "label": 1, "transform": name, "base": "imc25_test_fresh"})
    for i, t in enumerate(legit_pool[:OBF_LEGIT_N]):
        name = names[i % len(names)]
        obf_legit.append({
            "text": OBF_B7[name](t, random.Random(f"v7-obf-legit-{i}")),
            "label": 0, "transform": name, "base": "uci_test_ham_fresh"})

    # ---- held-out-family exam: toll + task_job fam_test rows ----
    hf_rows = []
    path = os.path.join(HERE, "data", "v4", "fam_test.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["family"] in HELD_OUT_FAMILIES:
            hf_rows.append({"text": r["text"], "label": 1,
                            "family": r["family"],
                            "template_id": r["template_id"]})

    # ---- hard legit train/val (guard: never collide with any
    # validation/test key of the frozen suites) ----
    guard = set(keys) | {t.strip().lower() for t in modern["test"]}
    legit_train, legit_val = build_legit(guard)

    def dump(name, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("recomb_train.jsonl", recomb_train)
    dump("recomb_validation.jsonl", recomb_val)
    dump("obf_eval_scam.jsonl", obf_scam)
    dump("obf_eval_legit.jsonl", obf_legit)
    dump("hf_eval.jsonl", hf_rows)
    dump("legit_train.jsonl", legit_train)
    dump("legit_validation.jsonl", legit_val)

    manifest = {
        "held_out_families": HELD_OUT_FAMILIES,
        "held_out_family_note": "toll + task_job excluded from ALL v7 "
                                "training and validation (family rows, "
                                "recombination parts). hf_eval = their "
                                "v4 fam_test rows. voter/jury_duty were "
                                "v6's exam (spent) and train normally.",
        "recomb_train_rows": len(recomb_train),
        "recomb_validation_rows": len(recomb_val),
        "recomb_pools": {"train": pool_train, "validation": pool_val},
        "obf_eval_B_v7": {
            "scam_rows": len(obf_scam), "legit_rows": len(obf_legit),
            "variants": names,
            "base_disjoint_from_v6_eval": True,
            "base": "IMC25 test positives / UCI test ham rows NOT used "
                    "by v6's eval (v6 selection reproduced and excluded);"
                    " fresh seeds v7-obf-*; built before v7 training; "
                    "evaluated once",
        },
        "hf_eval_rows": len(hf_rows),
        "legit_v7": {
            "train_rows": len(legit_train),
            "validation_rows": len(legit_val),
            "categories": [c for c, _ in LEGIT_TEMPLATES],
            "note": "deterministic slot-filled INFORM-ONLY templates; "
                    "split by template (index % 4 == 1 -> validation); "
                    "exact-match guarded against all frozen val/test "
                    "keys",
        },
        "generation": "deterministic code only (gen_v7_data.py, "
                      "obfuscate_v6.py, augment_v4.py); no LLM text",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
