#!/usr/bin/env python3
"""Deterministic obfuscation transforms for Scam Shield v6 (code only).

Two DISJOINT transform families, by design:

- TRAIN transforms (set A) — used to augment training data so the model
  learns scam *patterns* rather than exact surface forms:
  numeric leetspeak, inserted spaces, inserted punctuation, case flips,
  zero-width characters, and (legit side only) txt-speak shortening.

- EVAL transforms (set B) — used ONLY to build the held-out obfuscated
  evaluation set. These mechanisms NEVER appear in training:
  Cyrillic homoglyphs, symbol leetspeak (@/$/!), letter doubling,
  vowel dropping.

Keeping the sets disjoint is the point: the eval measures generalisation
to obfuscation styles the model was never shown, not memorisation of
the training transforms. All functions take a seeded random.Random.
"""
import random
import re

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

# ------------------------- TRAIN set A -------------------------

_LEET_DIGITS = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def leet_digits(text, rng, p_word=0.55):
    def rep(m):
        w = m.group(0)
        if len(w) < 3 or rng.random() > p_word:
            return w
        return "".join(_LEET_DIGITS.get(c.lower(), c)
                       if c.lower() in _LEET_DIGITS and rng.random() < 0.85
                       else c for c in w)
    return WORD_RE.sub(rep, text)


def space_split(text, rng, max_words=3):
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5
           and w.isalpha()]
    rng.shuffle(idx)
    for i in idx[:max_words]:
        w = words[i]
        cut = rng.randint(2, len(w) - 2)
        words[i] = w[:cut] + " " + w[cut:]
    return " ".join(words)


def punct_insert(text, rng, max_words=2):
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    rng.shuffle(idx)
    for i in idx[:max_words]:
        w = words[i]
        if rng.random() < 0.3 and len(w) <= 7:
            words[i] = rng.choice([".", "-"]).join(list(w))
        else:
            cut = rng.randint(1, len(w) - 1)
            words[i] = w[:cut] + rng.choice([".", "-", "."]) + w[cut:]
    return " ".join(words)


def case_flip(text, rng, max_words=3):
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    rng.shuffle(idx)
    for i in idx[:max_words]:
        words[i] = words[i].upper() if rng.random() < 0.6 else \
            "".join(c.upper() if j % 2 else c.lower()
                    for j, c in enumerate(words[i]))
    return " ".join(words)


def zero_width(text, rng, max_words=3):
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    rng.shuffle(idx)
    for i in idx[:max_words]:
        w = words[i]
        cut = rng.randint(1, len(w) - 1)
        words[i] = w[:cut] + "\u200b" + w[cut:]
    return " ".join(words)


_TXT_SPEAK = [("you", "u"), ("your", "ur"), ("are", "r"),
              ("because", "cuz"), ("tomorrow", "tmrw"), ("please", "pls"),
              ("thanks", "thx"), ("before", "b4"), ("see", "c"),
              ("great", "gr8"), ("tonight", "2nite")]


def txt_speak(text, rng):
    for a, b in _TXT_SPEAK:
        text = re.sub(rf"\b{a}\b", b, text, flags=re.IGNORECASE)
    return text


_TRAIN_SCAM = [leet_digits, space_split, punct_insert, case_flip,
               zero_width]
_TRAIN_LEGIT = [case_flip, space_split, punct_insert, txt_speak]


def obfuscate_train_scam(text, rng):
    """1 transform (70%) or 2 stacked (30%)."""
    fns = rng.sample(_TRAIN_SCAM, 2 if rng.random() < 0.3 else 1)
    for fn in fns:
        text = fn(text, rng)
    return text


def obfuscate_train_legit(text, rng):
    """Milder: a single light transform, so 'unusual formatting' alone
    never becomes a scam cue."""
    return rng.choice(_TRAIN_LEGIT)(text, rng)


# ------------------------- EVAL set B (never trained) -------------------------

_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р",
               "c": "с", "x": "х", "y": "у"}
_SYMBOL_LEET = {"a": "@", "s": "$", "i": "!", "g": "9"}


def homoglyph(text, rng, p_word=0.5):
    def rep(m):
        w = m.group(0)
        if len(w) < 3 or rng.random() > p_word:
            return w
        return "".join(_HOMOGLYPHS.get(c.lower(), c)
                       if c.lower() in _HOMOGLYPHS and rng.random() < 0.8
                       else c for c in w)
    return WORD_RE.sub(rep, text)


def symbol_leet(text, rng, p_word=0.55):
    def rep(m):
        w = m.group(0)
        if len(w) < 3 or rng.random() > p_word:
            return w
        return "".join(_SYMBOL_LEET.get(c.lower(), c)
                       if c.lower() in _SYMBOL_LEET and rng.random() < 0.85
                       else c for c in w)
    return WORD_RE.sub(rep, text)


def letter_repeat(text, rng, max_words=4):
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    rng.shuffle(idx)
    for i in idx[:max_words]:
        w = words[i]
        pos = rng.randint(0, len(w) - 1)
        words[i] = w[:pos] + w[pos] + w[pos:]
    return " ".join(words)


def vowel_drop(text, rng, max_words=4):
    words = text.split(" ")
    idx = [i for i, w in enumerate(words)
           if len(w) >= 4 and w.isalpha() and sum(
               c in "aeiou" for c in w.lower()) >= 2]
    rng.shuffle(idx)
    for i in idx[:max_words]:
        w = words[i]
        words[i] = w[0] + re.sub(r"[aeiou]", "", w[1:], flags=re.IGNORECASE)
    return " ".join(words)


EVAL_TRANSFORMS = {"homoglyph": homoglyph, "symbol_leet": symbol_leet,
                   "letter_repeat": letter_repeat, "vowel_drop": vowel_drop}
