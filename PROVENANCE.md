# Scam Shield — Baseline Provenance
Built: 2026-11-17/18 · Status: local baseline only — NOT published or uploaded anywhere

## Dataset
| Field | Entry |
|---|---|
| Name | SMS Spam Collection (UCI Machine Learning Repository, dataset 228) |
| Source URL | https://archive.ics.uci.edu/dataset/228/sms+spam+collection — zip downloaded directly from archive.ics.uci.edu (`data/sms.zip`) |
| Creators / citation | Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. "Contributions to the Study of SMS Spam Filtering: New Collection and Results." ACM DocEng 2011 |
| License | CC BY 4.0 (UCI Machine Learning Repository standard license; attribution required — given here). Verified 2026-11-18 via UCI dataset page and repository licensing |
| Contents | 5,574 labelled SMS (ham / spam) collected from free-for-research public sources per UCI: Grumbletext spam claims, NUS SMS Corpus ham (volunteers aware contributions would be public), Caroline Tagg PhD thesis ham, SMS Spam Corpus v.0.1 Big |
| Use here | `spam` treated as the positive (scam/spam) class, `ham` as legitimate. Exact-duplicate texts removed before splitting (5,574 → 5,159; 0 label conflicts) so no message appears in two splits |

## Frontier-model contact — affirmation
No Muse / Muse or OpenAI / Luna output was used as training data, labels, scores, rankings, filtering, or dataset selection for this model. Threshold selection was a fixed script rule applied to the validation split only (see `train.py`); the test split was evaluated once. No personal data was used.

## Method (pre-registered in train.py)
- Splits: train 70% (3,611) / validation 15% (774) / test 15% (774), stratified, seed 42
- Features: TF-IDF, word 1–2 grams, max 5,000 features, min_df 2, sublinear tf
- Classifier: Logistic Regression, C=4.0, class_weight=balanced, liblinear
- Threshold rule (validation only): highest scam recall among thresholds with validation false-positive rate on legitimate ≤ 1.0% → chosen threshold 0.56

## Held-out test results (evaluated once, threshold 0.56)
- Scam/spam recall: **0.9167** (88/96)
- False-positive rate on legitimate: **0.0015** (1/678) ← trust metric
- Precision (scam): 0.9888 · Accuracy: 0.9884
Full numbers: `metrics.json` (validation row included for transparency)

## Model files
- `model.npz` — 57,541 bytes (~0.055 MB, target <2 MB ✓): vocab (5,000), IDF, LR coefficients, intercept, threshold. Self-contained runtime in `predict.py` (pure NumPy reimplementation of the TF-IDF transform; parity vs sklearn verified, max probability difference 9.6e-08 on validation)
- `model_head.onnx` — 50,364 bytes: ONNX export of the logistic-regression head only (takes a float TF-IDF vector as input). Full-pipeline ONNX export via skl2onnx failed with a converter error; the .npz is the primary deployable

## Known limitations (honest)
- This dataset is 2011-era SMS *spam* (prizes, offers, marketing), not modern impersonation scams (grandparent / bank / gift-card scripts). It proves the pipeline and the false-positive discipline; it does NOT prove performance on 2026 scam texts. A modern, licensed smishing/scam corpus is the required next dataset before any real-world claim.
- Ham messages skew to Singaporean student SMS (NUS corpus); false-positive behaviour on seniors' real legitimate messages (bank alerts, appointment reminders, family texts) is untested and is the key risk to measure next.
- English only.

## Environment note
scikit-learn 1.9.1 and skl2onnx 1.20.0 were installed into `~/workspace/tools-venv` (via uv) for this build; training/inference run on CPU on this VM in seconds.

---

# v2 — modern smishing data added (2026-11-17/18)
Status: local iteration only — NOT published or uploaded anywhere. Script: `train_v2.py`; results: `metrics_v2.json`; model: `model_v2.npz`.

## Dataset added
| Field | Entry |
|---|---|
| Name | Smishing Dataset — IMC 2025 ("Fishing for Smishing", Agarwal, Papasavva, Suarez-Tangil & Vasek, ACM IMC '25) |
| Source URL | https://github.com/reportsmishing/Smishing-Dataset-IMC25 (`dataset/final_dataset_output.csv`, commit a617556), downloaded directly from the repo |
| License | **CC BY 4.0** — verified at the source repo 2026-11-18: `LICENSE.txt` present and README states "This work is licensed under a Creative Commons Attribution 4.0 International License" (attribution given here) |
| Contents | 33,869 user-reported smishing messages mined from five public reporting forums (reports spanning ~2019–2024, 66 languages), with scam_type and lure labels. Personal identifiers were masked by the researchers to placeholders (`<URL>`, `<PHONE_NUMBER>`, `<NAMED_ENTITY>`, `<DATE_TIME>`) |
| Use here | **English subset only**: 22,078 rows → exact-dedup, ≥15 chars, email-address screen (−6) → 15,762 unique positives. English scam mix: banking 10,495 · others 4,586 · government 1,922 · delivery 1,676 · telecom 1,630 · spam 1,387 · wrong-number 304 · "hey mum/dad" family-in-distress 77 (counts before dedup). All rows are reported smishing, so this set supplies modern positives only; legitimate negatives remain the UCI ham |
| Leakage guard | Any modern text also present in the UCI validation or test split was dropped (0 found). Modern split 70/15/15 (11,033 / 2,364 / 2,365), seed-42 shuffle |

## Datasets checked and EXCLUDED (not used)
| Dataset | Why excluded |
|---|---|
| Mishra & Soni SMS Phishing Dataset (Mendeley Data, DOI 10.17632/f45bkkt8pr.1; 5,971 msgs, ham/spam/smishing, CC BY 4.0 per the dataset's licence statements) | Not used: the original file could not be obtained directly from Mendeley in this session, and the Hugging Face copy (`ealvaradob/phishing-dataset`, card licence Apache-2.0) is a composite that merges the SMS with ~18k emails and 800k URLs under collapsed binary labels, losing the smishing class; its ham/spam also largely overlaps UCI (it builds on Almeida et al.). Revisit only with the original Mendeley file in hand |
| Ridham115/indian-scam-sms-synthetic-audited (HF) | Every row LLM-generated (Gemini) and LLM-audited; the card itself says not to use it to estimate real-world accuracy. Synthetic frontier-model data is out under this project's terms |
| useKled/scam-messages-you-received (HF) | Commercial licence by email request only; files are not openly downloadable — not a permissive public licence |
| TExtPhish/TExtPhish (HF) | CC BY-NC-ND 4.0, gated, extortion emails (not SMS) — NC/ND is not permissive |
| MOZNLP/MOZ-Smishing (HF) | Portuguese mobile-money messages (wrong language for this English model); samples also show unmasked personal names/phone numbers |
| SmishTank corpus (Timko & Rahman, CODASPY '24; 1,090 smishing) | Genuinely modern, but no licence for the dataset itself could be found at the source — unclear licence = excluded by rule |
| FredZhang7/all-scam-spam (HF) | Mixed-source aggregate including a ChatGPT-assisted subset; per-source licence/lineage unclear |
| bolewara/hinglish-scam-text-dataset (HF) | No licence stated on the card; mostly UCI-derived plus 63 hand-written rows — too small and unlicensed |

## Frontier-model contact — affirmation (v2)
Same as v1: no Muse / Muse or OpenAI / Luna output was used as training data, labels, scores, rankings, filtering, or dataset selection. Dataset screening above used only the datasets' own cards/papers and licences. Threshold selection was the same fixed script rule on validation only; each test below was evaluated once.

## Method (v2, pre-registered in train_v2.py)
- UCI splits reproduced exactly from v1 (same dedup, seed 42) so the v1 test rows are identical
- Training: UCI train split (3,611) + modern train positives (11,033) = 14,644
- Validation: UCI validation (774) + modern validation positives (2,364) = 3,138; threshold rule unchanged (max recall subject to legit FPR ≤ 1.0%) → chosen threshold **0.72**
- Identical model recipe to v1 (TF-IDF 1–2 g, 5,000 feats, LR C=4.0 balanced)

## Held-out test results (each evaluated once, threshold 0.72)
**Modern test** (2,365 held-out IMC25 English smishing vs 678 UCI test ham):
- Scam recall: **0.9569** (2,263/2,365) · Legit FPR: **0.0044** (3/678) · Precision 0.9987
- For scale, the **v1 model scores only 0.4516 recall** on these same modern positives — v1 missed over half of modern smishing

**v1 comparison** (identical UCI test split v1 was scored on):
- v2 scam recall: **0.7604** vs v1's 0.9167 — **worse on 2011-era spam**
- v2 legit FPR: **0.0044** vs v1's 0.0015 — slightly worse, still under the 1% rule

**Honest verdict:** v2 trades legacy marketing-spam recall for modern impersonation-smishing recall — a large gain where the real 2026 risk is (45% → 96%), at a real cost on the old distribution (92% → 76%). Neither number alone describes the model; which one matters depends on which messages it will actually see.

## Model file (v2)
- `model_v2.npz` — 62,670 bytes (~0.06 MB, target <2 MB ✓), same format as v1 (vocab, IDF, LR coefficients, intercept, threshold 0.72); `predict.py` loads `model.npz` (v1) — point it at `model_v2.npz` to run v2

## Known limitations (v2, honest)
- IMC25 texts are researcher-masked: real URLs/numbers were replaced by `<URL>` / `<PHONE_NUMBER>` placeholders, and texts were OCR-extracted from screenshots. Placeholder tokens are easy features, so the 95.7% modern recall is **optimistic** versus raw, unmasked real-world messages
- Exact-text dedup only: the same scam campaign reported many times with small variations can appear in both modern train and test, inflating modern test recall
- Modern positives skew to banking impersonation (about half of the English rows) and to scams people bothered to report — not a natural sample of all smishing
- Legit negatives are still 2011-era UCI ham (Singaporean student SMS); FPR on seniors' real legitimate bank/delivery texts remains unmeasured — still the key open risk
- English only

---

# v3 — combined training, balanced (2026-11-17/18)
Status: local iteration only — NOT published or uploaded anywhere. Script: `train_v3.py`; results: `metrics_v3.json`; model: `model_v3.npz`; runtime: `predict_v3.py` (pure-NumPy reimplementation of both TF-IDF branches; parity vs sklearn verified on validation, max probability difference 7.9e-08).

## Data (unchanged from v2 — no new source added)
Same two verified sources only: UCI SMS Spam Collection (CC BY 4.0) train split (ham + spam) and IMC25 (CC BY 4.0) English smishing positives, train split. **Legitimate negatives are UCI ham only — IMC25 contains no legitimate messages, and no other modern legitimate source was added.** All datasets excluded in the v2 section above remain excluded. Same splits as v1/v2 (seed 42), so test rows are identical across versions.

## What changed and why (recipe selected on VALIDATION only)
A validation-only sweep (12 recipes; test splits untouched during selection) diagnosed v2's UCI drop: 11,033 modern positives vs 449 UCI train spam, unweighted. Selection rule: maximize min(UCI, modern) validation recall, tie-break combined validation recall, threshold always by the fixed rule (max recall subject to legit FPR ≤ 1.0% on combined validation).
- **Group-balanced sample weights:** UCI ham, UCI spam, and IMC25 positives each contribute equal total training weight
- **Features:** word 1–2 g TF-IDF (15,000) ∪ char_wb 3–5 g TF-IDF (15,000), sublinear, min_df 2
- Same LR (C=4.0, class_weight balanced, liblinear). Chosen threshold **0.66** (validation FPR 0.0089)
- Validation at that threshold: UCI recall 0.9278, modern recall 0.9632 (v2 recipe on the same validation: UCI 0.8041, modern 0.9509)

## Held-out test results (threshold 0.66)
- **Modern IMC25 test recall: 0.9742** (2,304/2,365) — goal ≥90% ✓, beats v2's 0.9569
- **UCI test recall: 0.9479** (91/96) — recovered past v1's 0.9167, goal ~90% ✓
- **Legit FPR: 0.0029** (2/678 UCI test ham) — under the 1% rule ✓
- **Size: 310,698 bytes (~0.30 MB)** — under the 2 MB target ✓ (larger than v1/v2's ~60 KB because of the 30k-feature union)

## Comparison (full table in metrics_v3.json)
| Version | Modern recall | UCI recall | Legit FPR | Size |
|---|---|---|---|---|
| v1 (UCI only) | 0.4516 | 0.9167 | 0.0015 | 57,541 B |
| v2 (combined, unweighted) | 0.9569 | 0.7604 | 0.0044 | 62,670 B |
| v3 (combined, group-balanced, word+char) | **0.9742** | **0.9479** | 0.0029 | 310,698 B |

**Verdict: keep v3.** Combining did not hurt both — with source balancing it beat each single-distribution model on the other's home test. v1/v2 files retained for comparison.

## Frontier-model contact — affirmation (v3)
Same as v1/v2: no Muse / Muse or OpenAI / Luna output used as training data, labels, scores, rankings, filtering, tuning, or dataset selection. Recipe selection used validation metrics computed by script only.

## Honest notes (v3)
- Transparency: the first `train_v3.py` run crashed at the model-save step (a FeatureUnion attribute error) *after* test computation but before anything was printed or saved; no values from that run were seen or used to change any decision — the identical script was re-run after the save-step fix only
- All v2 limitations still apply: IMC25's masked placeholders (`<URL>`, `<PHONE_NUMBER>`) make modern recall optimistic vs raw real-world texts; exact-dedup only, so near-duplicate campaign variants can span modern train/test; legit negatives remain 2011-era UCI ham, so FPR on seniors' real bank/delivery/family texts is still unmeasured — still the key open risk before any real-world claim
- English only

---

# v4 — 2026 scam families + de-masking fix (2026-11-18)
Status: local iteration only — NOT published or uploaded anywhere. Scripts: `seeds_v4.json` (seed templates + sources), `augment_v4.py` (deterministic slot-filling + de-masking), `gen_v4_data.py` (dataset generation), `train_v4.py`; results: `metrics_v4.json`; model: `model_v4.npz`; runtime: `predict_v4.py`.

## What v4 set out to fix
1. v3 had never seen the newest 2026 scam families (toll, DMV ticket, IRS refund, task-job, jury-duty warrant, fake FTC agent, voter registration, WhatsApp code takeover, ...).
2. v3's known weakness: IMC25 masks real URLs/phones as `<URL>` / `<PHONE_NUMBER>` placeholders, so v3's 97.4% modern recall was optimistic versus raw real-world texts.

## New data — sources (ALL US government, public domain, fetched 2026-11-18)
Seed example texts come ONLY from these publications, or are faithful templates of the exact script mechanics they describe (each template's basis is recorded in `seeds_v4.json`). No social-media posts were used as data; no personal data; no LLM wrote any seed or training text.

| # | Source | Used for |
|---|---|---|
| 1 | FBI IC3 PSA I-041224-PSA (2024-04-12) — https://www.ic3.gov/PSA/2024/PSA240412 | toll (contains a verbatim published example text, used as seed toll-1) |
| 2 | FTC alert, "Got a text about unpaid tolls? It's probably a scam" (2025-01-17) — https://consumer.ftc.gov/consumer-alerts/2025/01/got-text-about-unpaid-tolls-its-probably-scam | toll |
| 3 | FTC alert, "That text about an overdue traffic ticket is probably a scam" (2025-07-21) — https://consumer.ftc.gov/consumer-alerts/2025/07/text-about-overdue-traffic-ticket-probably-scam | DMV ticket (threat list incl. "DMV violation database", 35% service fee, prosecution, credit score) |
| 4 | FTC alert, "That text or email about your 'tax refund' is a scam" (2026-01-22) — https://consumer.ftc.gov/consumer-alerts/2026/01/text-or-email-about-your-tax-refund-scam | IRS refund ("processed"/"approved" wording) |
| 5 | IRS IR-2026-30, "Dirty Dozen tax scams for 2026" (2026-03-05) — https://www.irs.gov/newsroom/dirty-dozen-tax-scams-for-2026-irs-reminds-taxpayers-to-watch-out-for-dangerous-threats | IRS impersonation by text (item 1) |
| 6 | FTC alert, "Did you get a call or text about a suspicious purchase on Amazon? It's a scam" (2024-03-07) — https://consumer.ftc.gov/consumer-alerts/2024/03/did-you-get-call-or-text-about-suspicious-purchase-amazon-its-scam | Amazon impersonation (incl. the v4-specified refund / "quality inspection, no return needed" variant, written as a template of this pattern) |
| 7 | FTC alert, "How to spot and avoid task scams" (2025-08) — https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams | task-job (optimization/product boosting, $5–$20 trust payouts, crypto deposit to unlock) |
| 8 | FTC alert, "That job offer text is probably a scam" (2026-04) — https://consumer.ftc.gov/consumer-alerts/2026/04/job-offer-text-probably-scam | task-job / fake recruiter (reply YES/INTERESTED, daily rates, deposit-check pattern) |
| 9 | FTC article, "Job Scams" — https://consumer.ftc.gov/articles/job-scams | fake-job equipment check (fake check deposit-and-send-back, starter kits/training fees) |
| 10 | FTC alert, "'When are you free for coffee?' Long lost friend or a scammer..." (2024-08-05) — https://consumer.ftc.gov/consumer-alerts/2024/08/when-are-you-free-coffee-long-lost-friend-or-scammer-trying-get-you-respond | wrong-number opener turning crypto |
| 11 | FTC alert, "Ignore calls, texts, and emails threatening to arrest you for missing jury duty" (2026-06-11) — https://consumer.ftc.gov/consumer-alerts/2026/06/ignore-calls-texts-and-emails-threatening-arrest-you-missing-jury-duty | jury duty / warrant (incl. texted fake warrant document twist) |
| 12 | FBI IC3 PSA I-091726-PSA (2026-09-17) — https://www.ic3.gov/PSA/2026/PSA260917 | government/law-enforcement impersonation (missed jury duty/court date + warrant pretense; payment via prepaid cards/crypto/courier) |
| 13 | FTC alert, "The FTC is not calling to deactivate your phone number" (2026-10-22) — https://consumer.ftc.gov/consumer-alerts/2026/10/ftc-not-calling-deactivate-your-phone-number | fake FTC agent (badge ID read aloud, deactivation threat, secrecy, security deposit) |
| 14 | FTC alert, "A real FTC employee won't text you their photo ID..." (2026-06-03) — https://consumer.ftc.gov/consumer-alerts/2026/06/real-ftc-employee-wont-text-you-their-photo-id-verify-their-identity | fake FTC agent (photo ID/badge, "recover losses") |
| 15 | FTC article, "Refund and Recovery Scams" — https://consumer.ftc.gov/articles/refund-and-recovery-scams | recovery scam ("get your money back for a fee": retainer/processing fee wording) |
| 16 | FBI IC3 PSA I-101526-3-PSA (2026-10-15) — https://www.ic3.gov/PSA/2026/PSA261015-3 | voter registration scheme ("not registered to vote" text → fake state registration page) |
| — | FTC alert, "The Google Voice scam: how this verification code scam works..." (2021-10) — https://consumer.ftc.gov/consumer-alerts/2021/10/google-voice-scam-how-verification-code-scam-works-how-avoid-it | WhatsApp verification-code takeover (code-forwarding pattern in FTC guidance; FTC rule: anyone who asks for your verification code is a scammer) |

## Augmentation (deterministic code only — `augment_v4.py`, seeded RNG)
- 46 seed templates across 11 families. Slots filled from fixed lists (amounts, dates, 35 states, toll agencies, counties, names); phone numbers are fictional 555 numbers; URLs are lookalike domains assembled by code patterns. Mechanical surface transforms only (prefix labels, trailing punctuation).
- Split BY TEMPLATE: per family, template index 1 → validation only, index 3 → test only, the rest → train. Generated rows: train 1,932 (24 templates), validation 456 (11 templates), test 538 (11 templates). The 2026-family test therefore measures generalisation to seed templates never trained on.
- De-masking: every IMC25 train text (11,033) was also added in a de-masked copy — all researcher placeholders (`<URL>`, `<PHONE_NUMBER>`, `<DATE_TIME>`, `<NAMED_ENTITY>`, `<LOCATION>`, `<EMAIL_ADDRESS>`, `<US_SSN>`, `<CREDIT_CARD>`, ...) replaced by code-generated raw forms (lookalike URLs, 555 numbers, dates, names/orgs from fixed lists; SSN-like strings use the invalid 900-series area). Same de-masker produces the raw-form stress test from held-out IMC25 test positives.

## Method (v4)
v3 recipe unchanged (word 1–2 g 15k ∪ char_wb 3–5 g 15k TF-IDF, LR C=4.0 balanced liblinear); sample weights extended to 5 balanced groups (UCI ham, UCI spam, IMC25 masked, IMC25 de-masked, 2026-family). Training rows 27,609. Threshold by the same fixed rule on combined validation (UCI + IMC25 + family validation) → **0.70** (validation FPR 0.0089; validation recalls: UCI 0.9278, modern 0.9670, family 0.9254).

## Held-out test results (each evaluated once, threshold 0.70)
- **Modern IMC25 test recall: 0.9772** (2,311/2,365) vs v3 0.9742 — slightly better
- **UCI test recall: 0.9479** (91/96) — identical to v3
- **Legit FPR: 0.0029** (2/678 UCI test ham) — identical to v3
- **2026-family held-out test recall: 0.9777** (526/538) on unseen seed templates. Per family: toll, DMV, IRS refund, Amazon, task-job, jury duty, FTC agent, job check, voter, WhatsApp code all **1.000**; **wrong-number 0.000 (0/12)** — see limitations
- **RAW-FORM stress test** (2,365 IMC25 test positives de-masked by code; identical rows for both models): **v3 recall drops 0.9742 → 0.9328 (−4.1 pts); v4 recall 0.9628 (−1.4 pts vs its masked score)** — the de-masking training data recovered most of the masking optimism
- **Size: 315,715 bytes (~0.30 MB)** — under the 2 MB target ✓

**Verdict: keep v4.** It matches or beats v3 on every old test, adds 10 of 11 new families at 100% on unseen templates, and cuts the raw-form penalty by about two-thirds. v3 files retained for comparison.

## Frontier-model contact — affirmation (v4)
Same as v1–v3: no Muse / Muse or OpenAI / Luna output was used as training data, labels, scores, rankings, filtering, or dataset selection. All new seed text is from the US-government publications listed above; all expansion was deterministic code. (The Amazon "quality inspection, no return needed" refund wording was a family variant specified in the v4 task brief and is templated on the FTC's Amazon-impersonation pattern, source 6 — flagged here for transparency.)

## Honest limitations (v4)
- **Wrong-number openers fail (0/12 on the held-out template).** The soft opener ("Oops, wrong person! ... want to keep chatting? I can show you how I earn money with cryptocurrency") scores below threshold: the training templates for this family were coffee/golf openers, and benign-looking openers genuinely resemble legitimate wrong-number texts. The explicit crypto-pitch variant does score 0.95. Flagging soft openers reliably would likely cost real false alarms on innocent wrong-number messages; reported as-is rather than tuned against the test.
- Slot-free seed templates yield few unique augmentations (test n = 11–14 for FTC-agent/job-check/WhatsApp-code/wrong-number vs 70 for slot-rich templates), so those per-family 1.000s rest on small samples.
- De-masked texts are code-generated raw forms, not real unmasked scam texts: the raw-form stress test measures robustness to placeholder removal, not true in-the-wild recall. Real raw smishing (with consent/provenance) remains the gold-standard test we do not have.
- The 2026-family test measures generalisation across templates written from the same government alerts — real scammer wording will vary more. Treat 0.9777 as "covers the documented scripts", not "catches 97.8% of 2026 scams".
- All v3 limitations still apply: legit negatives remain 2011-era UCI ham, so FPR on seniors' real bank/delivery/family texts is still unmeasured — still the key open risk. English only.

---

# v5 — transactional ask-vs-inform fix

## The flaw (diagnosed before fixing)
A parity probe of hand-written modern legitimate transactional texts
(`probe_legit.py` — EVAL ONLY, never training data) showed v4 flagging
**26 of 34 at ≥0.70 and 30 of 34 at ≥0.30** (mean probability 0.78):
a bare "Your verification code is 482916. Don't share it with anyone."
at 0.995, a Chase deposit alert at 0.998, a dentist reminder at 0.963.
On a held-out template split of generated transactional texts, v4's
legit false-positive rate was **53.0%** (151/285). Root cause: 2011 UCI
ham contains almost no modern transactional messages (2FA codes,
appointment/pharmacy/school reminders, tracking notices, balance
alerts), while scam positives do use code/verification language — so
the transactional register itself read as scam.

## New data (deterministic code only — `gen_v5_data.py`)
Programmatic slot-filled templates written for this fix; no LLM text,
no new external dataset. Two sides of one distinction:
- **Legit negatives (inform-only):** 2FA codes with "don't share it /
  we will never ask", appointment reminders (dental/medical/vet/salon),
  pharmacy refill-ready, school closure/bus notices, package tracking
  with no payment ask, bank balance/deposit/card-use alerts with no
  link or ask, utility/library/reservation/flight notices.
- **Scam positives (the ASK is the tell):** "send us the code we just
  texted you", "reply/share the verification code", "enter the code at
  <link>", appointment/pharmacy/tracking messages that demand a fee or
  card details via link, bank alerts asking for PIN/code.
- Rows: train 2,398 (1,701 legit / 697 scam), validation 493,
  held-out test 396 (285 legit / 111 scam). Split BY TEMPLATE within
  each category (index 1 → validation, index 3 → test), same
  discipline as v4. Two test categories rest on a single unique row
  because their held-out template has no slots (dedupe collapses its
  variants): the slot-free "password changed" notice is the one
  remaining legit FP, in both the test split and the probe.

## Method (v5)
v4 recipe unchanged; group balancing extended to 7 groups (adds
transactional legit and transactional scam). Training rows 30,007.
Threshold by the same fixed rule on combined validation → **0.77**
(validation FPR 0.0093).

## Held-out test results (each evaluated once, threshold 0.77)
- **Transactional held-out: legit FP rate 0.35% (1/285), vs v4's 53.0%
  on the identical rows; transactional scam recall 1.000 (111/111)**
- **Probe re-score: ≥0.70 count 26/34 → 1/34; ≥0.30 count 30/34 → 5/34;
  mean probability 0.78 → 0.11.** The parity case now scores 0.007
  (was 0.995); its scam counterpart ("send us the code…") scores 0.982
- Modern IMC25 test recall: **0.9674** (2,288/2,365) vs v4 0.9772
  (**−0.98 pt** — inside the 1-pt keep budget, but a real cost)
- UCI test recall: **0.9583** (92/96) vs v4 0.9479 (+1.0 pt)
- Legit FPR (UCI ham): **0.0029** (2/678) — identical to v4
- 2026-family held-out recall: **0.9758** vs v4 0.9777 (−0.2 pt)
- Raw-form stress recall: **0.9463** vs v4 0.9628 (**−1.7 pt** on the
  synthetic de-masked set — the second real cost of the fix)
- **Size: 316,854 bytes (~0.31 MB)** — under the 2 MB target ✓

**Verdict: keep v5.** The trade is explicit: ~1 pt of modern recall
and ~1.7 pts on the synthetic raw-form set, in exchange for fixing a
flaw that would have made the model cry wolf on essentially every
real 2FA text, appointment reminder, and delivery notice a senior
receives — the fastest possible way to make the target user ignore
it. v4 files retained for comparison.

## Frontier-model contact — affirmation (v5)
Same as v1–v4: no Muse / Muse or OpenAI / Luna output was used as
training data, labels, scores, rankings, filtering, or selection. The
v5 templates are programmatic code in `gen_v5_data.py`; the probe set
in `probe_legit.py` is hand-written evaluation material and was never
trained on.

## Honest limitations (v5)
- Password-changed / account-security notices remain the weakest
  legit category (the probe's one ≥threshold case, 0.845): they share
  wording with genuine account-takeover scam alerts.
- Transactional test coverage is template-based: real services phrase
  these messages in more ways than 68 templates. The probe (hand-written,
  independent of the templates) corroborates the fix, but a consented
  corpus of real transactional texts remains the gold standard we
  still do not have.
- Modern recall and raw-form recall each slipped ~1–1.7 pts (above);
  if a future version can recover them without reopening the
  transactional flaw, it should.
