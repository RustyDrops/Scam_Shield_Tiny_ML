# Scam Shield — Tiny ML

A micro scam-text classifier for seniors: paste a text message, get a plain-language verdict — **SAFE-LOOKING / SUSPICIOUS / LIKELY SCAM** — plus the words that triggered it. It runs fully offline, on-device, in 0.30 MB. No message ever needs to leave the phone.

Built as part of a micro edge-model pilot: models small enough to train and run on modest hardware, trained only on public data, with full provenance.

## Try it

Open `demo/index.html` in any browser — one self-contained file (548 KB), no network calls, no build step. Paste a message or use the built-in examples:

- Bank impersonation example → 99.9% LIKELY SCAM
- Delivery-fee example → 98.1% LIKELY SCAM
- Legitimate family message → 1.9% SAFE-LOOKING

The demo's JavaScript inference was verified against the Python model on 14 messages: maximum probability difference 3.4e-8.

## The model (v5 — current)

TF-IDF (word 1–2 grams + character 3–5 grams) + logistic regression, group-balanced training, threshold chosen on a validation split by a fixed rule (maximise recall subject to false-positive rate ≤ 1%); test splits were evaluated once.

| Version | Modern (2025) smishing recall | Legacy spam recall | False alarms on legitimate | Size |
|---|---|---|---|---|
| v1 (2011-era data only) | 45.2% | 91.7% | 0.15% | 57,541 B |
| v2 (modern data, unbalanced) | 95.7% | 76.0% | 0.44% | 62,670 B |
| v3 (combined, balanced) | 97.4% | 94.8% | 0.29% | 310,698 B |
| v4 (+ 2026 scam families, de-masked data) | 97.7% | 94.8% | 0.29% | 315,715 B |
| **v5 (+ transactional legit data)** | **96.7%** | **95.8%** | **0.29%** | **316,854 B** |

Full per-version detail: `metrics.json` … `metrics_v5.json`.

**v4 additions (2026):** trained on 46 seed templates covering 11 current scam families (unpaid toll/DMV, IRS refund, Amazon refund, task jobs, jury duty, fake FTC agent, recovery scams, and more), sourced from 17 FTC / FBI IC3 / IRS publications, expanded by deterministic code only — plus de-masked training data (real-looking links and numbers instead of placeholders). On a raw-form stress test, recall rose from v3's 93.3% to **96.3%**. Held-out 2026-family recall: **97.8%**. Known gap: soft "wrong number" openers that haven't pitched anything yet read as benign — flagging them would mean false-alarming innocent wrong-number texts.

**v5 fix (the important one):** v4 false-alarmed on **53%** of modern legitimate transactional texts — verification codes, appointment reminders, bank alerts — because its legitimate examples dated from 2011, before such texts existed. v5 adds code-generated transactional examples on both sides (a service *informing* you of a code = legitimate; a stranger *asking you to send* a code = scam). Transactional false-alarm rate: **0.35%**, transactional scam recall 100%, at a cost of ~1 point of modern recall (96.7%). Remaining weak spot: password-changed/security notices still tend to flag.

## Honest limits

- Modern recall is optimistic: the 2025 dataset masks URLs and phone numbers with placeholders, unlike raw real-world texts.
- False-alarm rate is measured on 2011-era legitimate messages, not seniors' real bank/delivery texts — that is the key unmeasured risk.
- This is a prototype aid, not a guarantee. When in doubt, call the bank/company on the number on your card — never the number in the message.

## Data & provenance

Training data is public only, each source license-verified at source:

- **UCI SMS Spam Collection** (Almeida et al., 2011) — CC BY 4.0
- **IMC 2025 Smishing Dataset** (Agarwal et al., ACM IMC '25) — CC BY 4.0

No AI-model output (Claude/Muse, OpenAI, or otherwise) was used as training data, labels, scores, or filtering at any stage. Several candidate datasets were *excluded* on license or provenance grounds (including one that turned out to be LLM-generated) — the full record, including exclusions, is in `PROVENANCE.md`.

## Files

- `model_v5.npz` — the v5 model (NumPy format); `predict_v5.py` — pure-NumPy inference (no sklearn needed at runtime)
- `train.py` … `train_v4.py` — training scripts; `gen_v4_data.py`, `augment_v4.py`, `seeds_v4.json` — 2026-family data generation; `gen_v5_data.py`, `probe_legit.py` — transactional data + the legit-probe that exposed v4's flaw
- `model.npz`, `model_v2.npz`, `model_v3.npz`, `model_v4.npz` — earlier versions, kept for comparison; `model_head.onnx` — v1 ONNX head experiment
- `demo/index.html` — the offline browser demo (v3 model embedded); `demo/build_demo.py` — its builder
- `PROVENANCE.md` — dataset licenses, exclusions, and per-version training record

## Run it (Python)

```bash
python3 predict_v5.py "Your bank account is locked, verify now at the link"
```

Datasets are not redistributed in this repo; the training scripts reference their public sources (see `PROVENANCE.md`).
