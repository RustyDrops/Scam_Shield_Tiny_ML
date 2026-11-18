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

---

# v6 — pattern-recognition experiment (2026-11-18) — NOT A KEEPER, v5 stays

Status: local experiment only — NOT published or uploaded anywhere.
Trained on Google Colab (CPU runtime, MCP route; two runs — the first
crashed on a missing bundle file, `predict_v4.py`, and was re-run
unchanged in recipe; account balance 597.73 → 597.45 units over the
whole window, shared account). Scripts: `obfuscate_v6.py`,
`gen_v6_data.py`, `train_v6.py`, `run_v6_colab.py`; results:
`metrics_v6.json`; new data under `data/v6/` (manifest there).

## What v6 set out to do
Move from memorising known scams to pattern recognition that catches
future threats: (1) wider character features (char_wb 3–6 g, 35k,
plus word 1–2 g 15k); (2) deterministic obfuscation augmentation —
set A transforms (numeric leet, space splits, punctuation inserts,
case flips, zero-width chars; txt-speak for legit copies) applied to
4,500 scam + 900 legit training copies; (3) 2,600 novel recombinations
(openers/bodies/CTAs mixed across seed families — texts that never
existed as seeds); (4) two seed families (voter, jury_duty) excluded
from ALL training and validation data as a "future scam" test.

## New held-out evaluations (built before training, evaluated once)
- **Obfuscated set** (800 IMC25 test positives + 300 UCI test ham,
  transformed with set B mechanisms NEVER used in training: Cyrillic
  homoglyphs, symbol leet @/$/!, letter doubling, vowel dropping):
  **v6 recall 0.9537 vs v5 0.9375 (+1.6 pts)** — all four transforms
  improved (homoglyph +2.0, vowel drop +2.0, symbol leet +1.5, letter
  repeat +1.0); obfuscated-legit FP 0.67% vs v5 1.0%.
- **Held-out-family set** (voter + jury_duty fam_test rows, 140):
  **v6 0.9571 vs v5 0.9929 (−3.6 pts)**. Voter 1.000 for both;
  jury_duty v6 0.9143 vs v5 0.9857. Structural caveat: v5 trained on
  sibling templates of both families, so its baseline measures
  template generalisation; v6's 95.7% on wholly unseen families is
  the honest "future family" number — but the pre-registered keeper
  rule demanded improvement over v5, and v6 does not deliver it here.

## Frozen regression suites (each evaluated once, threshold 0.66)
- Modern IMC25 recall **0.9746** (v5 0.9674, +0.7 pt) ✓ gate ≥0.96
- Legit FPR **0.0059 (4/678)** vs v5 0.0029 — **✗ breaches the ≤0.5%
  gate** (two extra false alarms on 2011 ham)
- UCI recall 0.9583 (identical to v5); 2026-family overall 0.9665
  (v5 0.9758 — includes the two deliberately untrained families)
- Raw-form stress **0.9569** (v5 0.9463, +1.1 pts)
- Transactional: legit FP 0.0035, scam recall 1.000 — identical to v5 ✓
- Probe: mean prob 0.087 (v5 0.105), 1/34 ≥ threshold — same as v5
- Size: 480,211 bytes (~0.46 MB) — under the 2 MB target ✓

**Verdict: NOT a keeper — v5 remains the keeper.** The pre-registered
rule required no regression (modern ≥96% ✓, FPR ≤0.5% ✗, txn FP ≤1%
✓) AND clear improvement on both new sets (obfuscated +1.6 pts —
real but under the +2 pt bar; held-out-family −3.6 pts ✗). v6's
candidate model file was therefore not retained as a deliverable.
What v6 proved: unseen-obfuscation robustness and unseen-family
transfer (95.7% combined, voter perfect) are achievable at this size;
what it cost: a doubled ham FPR at the validation-chosen threshold.
A future v7 could try to recover the FPR with a stricter validation
FPR cap — that would be a new pre-registered run, not a re-tune of
this one (all v6 test numbers above are spent).

## Frontier-model contact — affirmation (v6)
Same as v1–v5: no Muse / Muse or OpenAI / Luna output was used as
training data, labels, scores, rankings, filtering, or selection.
All new text is deterministic code: seed-template recombination and
mechanical character transforms. Training ran on Colab as compute
only.

---

# v7 — keep v6's pattern wins, fix the false-alarm rate (2026-11-18)

Status: PRE-REGISTERED — this section's design and gates were written
BEFORE any v7 training or evaluation. Results and verdict are appended
below the line once the single run completes. Scripts: `gen_v7_data.py`,
`train_v7.py`, `run_v7_colab.py`; results: `metrics_v7.json`; new data
under `data/v7/` (manifest there).

## Design (fixed in advance)
Recipe = v6's, unchanged in its wins: word 1–2 g (15k) + char_wb 3–6 g
(35k) TF-IDF, LR C=4.0, group-balanced weights, set-A obfuscation
augmentation (4,500 scam + 900 legit copies), cross-family seed
recombinations. Two changes, both aimed at v6's one failure (test legit
FPR 0.59% vs the ≤0.5% gate):

1. **Stricter threshold discipline.** The operating threshold is chosen
   on VALIDATION ONLY, maximising scam recall subject to a HARD
   constraint: validation legit FPR ≤ 0.30% (v6 used a 1.0% cap and
   landed at 0.82% validation / 0.59% test). The 0.30% cap is the
   pre-registered margin under the 0.5% test gate. If no grid threshold
   satisfies the cap, the threshold with the lowest validation FPR is
   used and that fact is recorded.
2. **More hard legit training data.** A new deterministic template set
   (`gen_v7_data.py`, v5 transactional style: slot-filled code only)
   adds legit informs — 2FA/security codes, appointment and delivery
   reminders, bank/school/utility informs, short casual texts — as an
   extra training group, plus a template-disjoint validation slice so
   the stricter threshold rule sees this text style too.

## Fresh held-out sets (built before training, each evaluated ONCE)
- **Obfuscation set B-v7**: IMC25 test positives and UCI test ham NOT
  used by v6's eval (disjoint base rows, fresh seeds), transformed with
  the same never-trained set-B mechanisms (homoglyph, symbol leet,
  letter repeat, vowel drop) at fresh parameter variants, including
  some stacked pairs. v5 is scored on it once as the baseline.
- **Held-out families**: **toll** and **task_job** are excluded from
  ALL v7 training and validation (family rows and recombination parts).
  voter/jury_duty were v6's exam and are spent — they return to
  training. The exam is their v4 fam_test rows; v5 is scored on them
  once as the baseline (v5 trained on sibling templates of both, so its
  baseline is the template-generalisation number, as in v6).
- **Frozen v5 suites** (modern, UCI, 2026-family, raw-form stress,
  transactional, probe) re-evaluated once as regression.

## Keeper gates (ALL must pass, fixed in advance)
1. Modern recall ≥ 0.965
2. Legit FPR ≤ 0.005 (modern test block)
3. Transactional legit FP ≤ 0.01
4. Obfuscated-B-v7 recall STRICTLY greater than v5's on the same set
5. Held-out-family (toll + task_job) recall ≥ v5's on the same families

If any gate fails: v7 is rejected, v5 stays the keeper, `model_v7.npz`
and `predict_v7.py` are NOT retained, and there is no re-tuning against
these spent tests.

## Budget
Colab cap for this run: 4 compute units (project authorisation 20,
~0.7 used by the earlier v6/RA/WIR runs). CPU runtime. Runtime deleted
in a finally block; assignments verified empty afterwards.

## Frontier-model contact — affirmation (v7)
Same as v1–v6: no Muse / Muse or OpenAI / Luna output is used as
training data, labels, scores, rankings, filtering, or selection. All
new text is deterministic code. Training runs on Colab as compute only.

## Results (appended after the single run)

Run: Google Colab CPU runtime, MCP route, single run of the
pre-registered recipe exactly as written above (`gen_v7_data.py` data
verified row-for-row against its manifest; `train_v7.py`;
`run_v7_colab.py`). Account balance 595.06 → 595.06 units (0.00 at
displayed precision; ~4-minute CPU session, cap was 4). Runtime
deleted in a finally block; assignments verified empty afterwards.
Threshold chosen on validation: **0.78**, and the pre-registered cap
was satisfied outright — validation legit FPR **0.0026 ≤ 0.0030**
(val legit n=1,160, incl. the 189 new legit-v7 rows; validation scam
recall 0.9482). No fallback was needed.

### Frozen regression suites (each evaluated once, threshold 0.78)
- Modern IMC25 recall **0.9619** (2,275/2,365) vs v5 0.9674 — ✗ gate
  ≥0.965 missed by 0.3 pt
- Legit FPR **0.0029** (2/678) vs v5 0.0029 — ✓ gate ≤0.005
- UCI recall 0.9479 (v5 0.9583); raw-form stress 0.9311 (v5 0.9463)
- Transactional: legit FP **0.0000** (0/285, v5 0.0035) — ✓ gate
  ≤0.01; transactional scam recall 0.991 (v5 1.000)
- Probe: mean prob 0.052 (v5 0.105), **0/34 ≥ threshold** (v5 1/34) —
  the cleanest legit behaviour of any version
- 2026-family overall 0.9145 — dragged down by the deliberately
  untrained task_job (below); every trained family is 1.000 except
  jury_duty 0.9429 and wrong_number 0.000 (known, untuned)

### Fresh held-out sets (evaluated once; v5 scored once as baseline)
- **Obfuscation set B-v7** (800 scam / 300 legit, disjoint bases,
  fresh variants incl. stacked pairs): v7 recall **0.9062** vs v5
  **0.9125** (−0.6 pt) — ✗ gate (needed strictly greater). Mixed per
  transform: stacked homoglyph+vowel_drop slightly better for v7
  (+0.7), homoglyph and letter-repeat tied, symbol-leet variants
  worse (−0.8 to −2.3). Obfuscated-legit FP identical: 0.33% both.
  Unlike v6's set, B-v7 gives v7 no obfuscation edge at all.
- **Held-out families** (toll + task_job fam_test, 140 rows): v7
  **0.7857** vs v5 **1.0000** — ✗ gate. Toll: 1.000 for both (v7
  catches a family it never trained on — toll language transfers).
  **task_job: v7 0.5714 vs v5 1.0000** — the run's big miss. As
  pre-registered, v5's baseline is template generalisation (it
  trained on sibling task_job templates); v7's 57% is the honest
  never-seen-family number, and it says task-scam language
  (optimisation tasks, small trust payouts, crypto unlock deposits)
  does NOT transfer from the other families and recombinations the
  way toll language does.

### Keeper gates
| Gate | Result |
|---|---|
| Modern recall ≥ 0.965 | ✗ 0.9619 |
| Legit FPR ≤ 0.005 | ✓ 0.0029 |
| Transactional legit FP ≤ 0.01 | ✓ 0.0000 |
| Obfuscated-B-v7 recall > v5 | ✗ 0.9062 vs 0.9125 |
| Held-out-family recall ≥ v5 | ✗ 0.7857 vs 1.0000 |

**Verdict: NOT a keeper — v5 remains the keeper.** Per the
pre-registered rule, `model_v7.npz` and `predict_v7.py` are NOT
retained and there is no re-tuning against these spent tests. The
FPR half of the design worked exactly as registered: the 0.30%
validation cap held on test (0.29%), transactional false alarms went
to zero, and the hand-written probe is cleaner than v5's. What it
cost: the higher threshold plus the legit-heavy training shaved
recall everywhere (modern −0.6 pt vs v5, raw-form −1.5 pts), erased
v6's obfuscation edge, and the task_job exam exposed a genuine
unseen-family transfer gap that v6's voter/jury_duty exam did not.
Size would have been 481,902 bytes (~0.46 MB). Full numbers:
`metrics_v7.json`.

---

# v8 — midpoint threshold cap + task-scam-space training families (2026-11-18)

Status: PRE-REGISTERED — this section's design and gates were written
BEFORE any v8 training or evaluation. Results and verdict are appended
below the line once the single run completes. Scripts: `gen_v8_data.py`,
`train_v8.py`, `run_v8_colab.py`; seeds: `seeds_v8.json`; results:
`metrics_v8.json`; new data under `data/v8/` (manifest there).

## What v7 taught (fixed inputs to this design)
- v7's false-alarm fix worked perfectly: test legit FPR 0.0029,
  transactional legit FP 0.0000, cleanest probe yet (mean prob 0.052,
  0/34 ≥ threshold). The hard-legit group (gen_v7_data) and the
  template-disjoint legit validation slice are CREDITED and KEPT
  unchanged in v8.
- v7's strict 0.30% validation cap overpaid for that safety: chosen
  threshold 0.78 cost recall everywhere (modern 0.9619 vs the 0.965
  gate; raw-form 0.9311 vs v5 0.9463) and erased the obfuscation edge
  on fresh set B-v7 (0.9062 vs v5 0.9125).
- v7's held-out exam exposed a real transfer gap: never-seen
  **task_job scored 0.5714** while toll scored 1.0000. Task/gig/
  crypto-unlock scam language does NOT transfer from the other
  families and recombinations.

## Design (fixed in advance)
Recipe = v7's, unchanged except the three numbered changes below:
word 1–2 g (15k) + char_wb 3–6 g (35k) TF-IDF, LR C=4.0,
group-balanced weights, set-A obfuscation augmentation (4,500 scam +
900 legit copies), cross-family recombinations, v7's hard-legit
training group AND its template-disjoint validation slice.

1. **Midpoint threshold cap.** The operating threshold is chosen on
   VALIDATION ONLY, maximising scam recall subject to validation
   legit FPR ≤ **0.0040** — the pre-registered midpoint hypothesis
   between v6's loose 1.0% cap (test FPR landed 0.59%, gate breach)
   and v7's strict 0.30% (recall overpaid). If no grid threshold
   satisfies the cap, the threshold with the lowest validation FPR is
   used and that fact is recorded.
2. **New TRAINING families in the task-scam space** (`seeds_v8.json`,
   built the v4 way: deterministic slot-filled templates via
   augment_v4, split by template — index 1 validation-only, index 3
   test-only, the rest train). Two families, deliberately adjacent to
   task_job's mechanics but with their own templates and vocabulary:
   - **gig_task** — app/gig task work: product-boosting and
     review/liking task sets in an app or chat group, small early
     payout, level-ups, premium/VIP sets, negative-balance top-ups.
   - **crypto_unlock** — fake exchange/wallet balances that can only
     be released after a "verification deposit", withdrawal "tax" or
     "unlock fee" paid in crypto; frozen-account unlock demands.
   Sources (US-government consumer publications, public domain;
   templates are faithful slot-templates of the script mechanics
   these publications describe — no text copied, no LLM text):
   - FTC Data Spotlight, "Paying to Get Paid: Gamified Job Scams
     Drive Record Losses" (Dec 2024), ftc.gov — task sets, small
     early payouts, deposit-to-unlock mechanics, ~20,000 task-scam
     reports in H1 2024.
   - FTC Consumer Alert, "How to spot and avoid task scams"
     (2025-08-14), consumer.ftc.gov.
   - FTC Consumer Advice, "Job Scams", consumer.ftc.gov.
   - FBI, "Cryptocurrency Investment Fraud", fbi.gov — incl. the
     "Cryptocurrency Job Scams" section: account frozen at
     withdrawal, arbitrary "taxes"/"fees" demanded to unlock funds.
   - FinCEN Alert on pig-butchering / crypto investment fraud
     (Sep 2023) — small early withdrawals allowed, then "taxes,
     fees or penalties" demanded before withdrawal.
   These families join training AND recombinations (train-split
   template parts only). task_job and toll (v7's spent exam) return
   to normal training, as voter/jury_duty did in v7.
3. **Fresh held-out exam families** (`seeds_v8.json`, same file,
   built the same v4 way): **romance** and **grandparent** — never
   used as an exam before (voter/jury_duty were v6's; toll/task_job
   v7's — all spent). NOTE: contrary to the run brief's assumption,
   romance/grandparent do NOT exist in seeds_v4.json (its 11 families
   are toll, dmv, irs_refund, amazon, task_job, wrong_number,
   jury_duty, ftc_agent, job_check, voter, whatsapp_code), so they
   are authored NEW here in seeds_v8.json. They are excluded from
   ALL v8 training and validation — family rows AND recombination
   parts. The exam is generated variants of ALL their templates
   (70 per family, v4 test-variant count). Because the families are
   new in v8, v5 never trained on them either: v5's baseline on this
   exam is a true never-seen-family score, not template
   generalisation — a cleaner exam than v6/v7's. Sources:
   - FTC Consumer Advice, "What To Know About Romance Scams",
     consumer.ftc.gov — fake profiles, can never meet (military /
     oil rig / abroad), money for medical bills, plane tickets,
     visas, customs fees; crypto-investing offers; payment by wire,
     gift-card PINs, payment apps, crypto.
   - FTC Consumer Advice, "Family emergency scams",
     consumer.ftc.gov, and FCC Consumer Guide, "'Grandparent'
     Scams" — grandchild in an accident / arrested / hospitalised,
     "don't tell Mom and Dad", a "lawyer" or bondsman takes over,
     urgent wire / gift cards / courier cash pickup.
   - FBI IC3 PSA I-120324-PSA (2024-12-03), "Criminals Use
     Generative Artificial Intelligence to Facilitate Financial
     Fraud" — voice-cloned relative in a crisis demanding immediate
     money (the text-message form of the same script is used here).

## Fresh held-out sets (built before training, each evaluated ONCE)
- **Obfuscation set C**: base rows disjoint from BOTH v6's eval and
  B-v7 (both selections reproduced exactly and excluded). What
  remains of the seed-42 held-out pools is **765 IMC25 test
  positives and 78 UCI test ham** — set C uses ALL of them (the
  smaller legit n is a property of the remaining pool, recorded
  here in advance; legit FP on set C is reported, not gated).
  Transforms: the same never-trained set-B mechanisms at the B-v7
  parameter variants (homoglyph_hi, symbol_leet_hi,
  letter_repeat_hi, vowel_drop_hi + the two stacked pairs), fresh
  seeds (`v8-obf-*`). v5 is scored on it once as the baseline.
- **Held-out families**: romance + grandparent exam rows (above).
  v5 scored once as the baseline.
- **Frozen v5 suites** (modern, UCI, 2026-family, raw-form stress,
  transactional, probe) re-evaluated once as regression. The
  2026-family suite additionally reports the new training families'
  test-split rows (gig_task / crypto_unlock template index 3).

## Keeper gates (ALL must pass, fixed in advance)
1. Modern recall ≥ 0.965
2. Legit FPR ≤ 0.005 (modern test block)
3. Transactional legit FP ≤ 0.01
4. Obfuscation-C recall STRICTLY greater than v5's on set C
5. Held-out-family (romance + grandparent) recall ≥ v5's on the
   same families

If any gate fails: v8 is rejected, v5 stays the keeper, `model_v8.npz`
and `predict_v8.py` are NOT retained, and there is no re-tuning
against these spent tests.

## Budget
Colab cap for this run: 4 compute units (project authorisation 20).
CPU runtime. Runtime deleted in a finally block; assignments verified
empty afterwards. Launch precondition: no other job's runtime active
(Read-Aloud v3 / Write It Right v5 share the queue).

## Frontier-model contact — affirmation (v8)
Same as v1–v7: no Muse / Muse or OpenAI / Luna output is used as
training data, labels, scores, rankings, filtering, or selection. All
new text is deterministic code: seed templates are slot-filled by
augment_v4 from fixed lists. Training runs on Colab as compute only.
(Web sources above were read by the operator to ground the template
mechanics in the cited government publications, the same standard as
seeds_v4.json.)

## Results (appended after the single run)

**Verdict: v8 REJECTED — 2 of 5 gates. v5 stays the keeper.**
`model_v8.npz` was pulled back, SHA-256 verified
(fbfeca88…7c1, 482,176 B), then deleted per the rule above;
`predict_v8.py` was never written. Numbers in `metrics_v8.json`.

Run facts: Colab CPU runtime created in parallel with Read-Aloud
v3's T4 GPU job (Rusty's 2026-11-18 concurrency rule — CPU jobs no
longer queue behind GPU jobs); total driver time 236 s; runtime
deleted in the finally block and verified gone (only the GPU
runtime remained afterwards, untouched). Balance 593.42 → 593.32
over the whole working window — that 0.10 delta is shared with the
concurrent T4 job (1.15 units/hr); v8's own CPU share at
~0.08 units/hr for a ~4-minute runtime is ≈0.00–0.01 units, far
under the 4-unit cap. One audit fix before launch: the prepped
driver's spec list fell back to L4/T4 GPUs if CPU assignment failed
— removed (CPU std → CPU highmem only), since a GPU fallback would
have breached both the budget design and GPU serialization.

Threshold: the relaxed 0.40% cap WAS satisfied on validation
(chosen 0.81, val FPR 0.0034, val recall 0.9425, val legit n=1,160)
— but the chosen threshold came out HIGHER than v7's 0.78, so the
midpoint hypothesis bought nothing: the binding constraint is the
validation legit distribution, not the cap value. Test legit FPR
0.0029 ✓, transactional legit FP 0.0000 with transactional scam
recall 1.0000 ✓, and the cleanest probe yet (mean prob 0.049,
0/34 ≥ threshold). The safety side of the v7/v8 line is solved;
recall is the whole remaining problem.

| Gate | v8 | v5 | Pass |
|---|---|---|---|
| Modern recall ≥ 0.965 | 0.9543 | 0.9674 | ✗ |
| Legit FPR ≤ 0.005 | 0.0029 | 0.0029 | ✓ |
| Transactional FP ≤ 0.01 | 0.0000 | 0.0035 | ✓ |
| Obfuscation-C recall > v5 | 0.8902 | 0.9137 | ✗ |
| Held-out-family recall ≥ v5 | 0.1429 | 0.2071 | ✗ |

What the run taught:
- **The new training families were learned** — on their own
  template-disjoint test splits: gig_task 1.0000 (n=70),
  crypto_unlock 0.9714 (n=70). The original gap family task_job
  improved to 0.7429 on its test split (v7's never-seen exam had it
  at 0.5714) — better, still the weakest trained family bar
  jury_duty (0.4857, a regression vs v7's 0.9429 on the same suite)
  and the deliberately untuned wrong_number (0.0, n=12).
- **The fresh exam is a different register, and both models fail
  it.** Romance + grandparent scam language is conversational and
  emotional — few URLs, amounts, or scam keywords — and largely
  evades a TF-IDF model entirely: v8 0.1429, v5 0.2071 (romance:
  0.0857 vs 0.2000; grandparent: 0.2000 vs 0.2143). v5's edge here
  is real but both scores are poor in absolute terms; this exam
  measures a capability neither model has, not a tuning gap.
- Modern recall fell for the third straight round (v5 0.9674 →
  v7 0.9619 → v8 0.9543) and raw-form fell with it (v8 0.9205 vs
  v5 0.9463); UCI recall 0.9375 (v5 0.9583). Obfuscation-C: v8
  trailed v5 on every one of the six transforms. Adding groups and
  raising the effective threshold keeps trading headline recall
  for safety the keeper already had.
- Obfuscation set C legit FP was 0.0000 for BOTH models (n=78, as
  pre-registered the small remaining pool; reported, not gated).

Standing conclusion after v6–v8: three consecutive candidates have
failed against v5, each for a different diagnosed reason, and the
v5 recipe's balance (threshold 0.77, no hard-legit group, no cap
tuning) has not been beaten. The next lever, if any, is not another
threshold or group change — the romance/grandparent result points
at conversational-scam training data as the missing capability,
and the modern-recall slide points at simplifying back toward the
v5 recipe rather than adding to it. No v9 is pre-registered here;
that is a fresh design decision.

---

# v9 — back to the v5 recipe + conversational-scam data (2026-11-18)

Status: PRE-REGISTERED — this section's design and gates were written
BEFORE any v9 data generation, training, or evaluation. Results and
verdict are appended below the line once the single run completes.
Scripts: `seeds_v9.json`, `gen_v9_data.py`, `train_v9.py`,
`run_v9_colab.py`; results: `metrics_v9.json`; new data under
`data/v9/` (manifest there).

## What v6–v8 taught (fixed inputs to this design)
- **Threshold-cap games are exhausted.** v7's 0.30% cap chose 0.78,
  v8's 0.40% cap chose 0.81 — both HIGHER than v5's 0.77 — and modern
  recall fell three rounds straight: v5 0.9674 → v7 0.9619 → v8
  0.9543. The binding constraint is the validation legit
  distribution, not the cap value. v9 therefore uses **v5's own
  threshold rule** (max recall subject to validation legit FPR
  ≤ 1.0%, grid 0.05–0.95 step 0.01) with NO cap variant. Safety
  valve only: if no grid threshold satisfies the rule, the
  lowest-validation-FPR threshold is used and that fact is
  recorded (v5's run did not need it).
- **The v8 fresh exam exposed a DATA gap, not a tuning gap.**
  Conversational/emotional scams (romance, grandparent) scored
  0.1429 (v8) / 0.2071 (v5): few URLs, amounts, or scam keywords —
  nothing for the current features to grip. The fix is training
  data in that register, plus matching legitimate conversational
  data so the model learns conversational ≠ scam. For this
  audience (seniors), the false-alarm side is the whole ballgame.
- The v6–v8 additions (char 3–6 g 35k features, set-A obfuscation
  copies, cross-family recombinations, v7 hard-legit group) are
  NOT carried into v9. v9 tests one hypothesis: v5's exact recipe
  + conversational data.

## Design (fixed in advance)
**Base = v5, exactly.** Feature config: word 1–2 g TF-IDF (15k,
min_df 2, sublinear, v5 token pattern) ∪ char_wb 3–5 g TF-IDF
(15k, min_df 2, sublinear); LR C=4.0, class_weight balanced,
liblinear. Training data = v5's exact seven groups (UCI ham, UCI
spam, IMC25 masked, IMC25 de-masked, v4 2026-family, transactional
legit, transactional scam) with group-balanced sample weights
extended from 7 to 9 groups by the two added groups below.
Validation = v5's combined validation plus the added groups'
validation slices. Threshold rule = v5's own rule (above).

**Added training data — the ONLY change from v5** (all
deterministic code, v4-style slot-filled templates via augment_v4;
sources under "Added-data sources" below):

1. **Conversational scam group** (one balanced group):
   - **romance** and **grandparent** — v8's exam families, now
     spent as an exam and eligible for training. Templates are the
     seeds_v8.json families, unchanged, split BY TEMPLATE with the
     v4 rule (index 1 → validation only, index 3 → internal test
     only, the rest → train); variant counts 90/50/70 per template
     (the v8 fam8 convention), fresh v9 seeds.
   - **wrong_number_opener** — NEW family in seeds_v9.json:
     wrong-number conversational openers that pivot to a crypto /
     mentor / investment pitch (v4's known deliberate gap: its
     wrong_number held-out template scored 0/12 in v4–v8 and was
     left untuned). Same template split rule and variant counts.
2. **Conversational legit group** (one balanced group,
   `gen_v9_data.py`, v7 template style: slot-filled code only):
   genuine wrong-number replies, wrong-person apologies,
   family/friendly chat-style messages. INFORMAL ONLY — no links,
   no asks for money, codes, or personal details (a small number of
   ordinary family money mentions, e.g. birthday cash to a named
   relative, are included deliberately: family talking about money
   is not a scam tell). Split BY TEMPLATE within each category
   (index % 4 == 1 → validation, else train; 50 variants per train
   template, 30 per validation template), exact-match guarded
   against all frozen validation/test keys. Volume target ≈1,500
   train / ≈300 validation rows — comparable to the v5
   transactional legit group (1,701 train).

## Added-data sources (US government, public domain)
Templates are faithful slot-templates of the script mechanics these
publications describe — no text copied, no LLM text:
- romance / grandparent: the sources already recorded in
  seeds_v8.json (FTC "What To Know About Romance Scams"; FTC
  "Family emergency scams"; FCC "'Grandparent' Scams" guide; FBI
  IC3 PSA I-120324-PSA, 2024-12-03).
- wrong_number_opener: FTC Consumer Alert, "'When are you free for
  coffee?' Long lost friend or a scammer trying to get you to
  respond" (2024-08-05), consumer.ftc.gov (already v4 source #10);
  FinCEN Alert on pig-butchering / crypto investment fraud
  (Sep 2023) — wrong-number initial contact grooming into crypto
  investment; FBI IC3 PSA I-120324-PSA (2024-12-03).
- charity_relief (exam): FTC Consumer Advice, "When donating,
  support those in need, not a scammer" (consumer.ftc.gov, 2026)
  and FTC, "Charity Scams After a Disaster" (ftc.gov) — fake
  charities and impersonated real ones after disasters; pressure
  to give now; payment only by gift card / wire / crypto.
- package_delivery (exam): FTC Consumer Alert, "Think that text
  message is from USPS? It could be a scam" (2025-04-23),
  consumer.ftc.gov; FTC Data Spotlight, "Top Text Message Scams of
  2024" (Apr 2025) — fake package-delivery problems the most
  reported text scam of 2024; U.S. Postal Inspection Service,
  "Smishing: Package Tracking Text Scams" (uspis.gov) — USPS texts
  only on request, never with a link, never demanding a redelivery
  fee.

## Exam-family verification (charity_relief + package_delivery)
Checked against this file, seeds_v4.json, seeds_v8.json, and all
gen/train scripts before registration: **"charity" appears nowhere
in any prior round** — charity_relief is a genuinely new family.
**package_delivery has never existed as a seed-template family**
either; DISCLOSED OVERLAP: delivery *register* is partially seen —
v5's transactional group contains package tracking informs and
redelivery-fee scam asks, and IMC25's modern set includes a
delivery scam_type (≈1.7k rows pre-dedup). The exam templates are
authored NEW, emphasising mechanics the transactional templates do
not use (address-incomplete correction links, customs clearance
fees, recipient-name mismatch, storage fees). The gate compares v9
and v5 on identical rows, so the comparison is fair; the exam is
recorded as "new family, partially seen register" — weaker than
v8's romance/grandparent exam, and the results will be read that
way.

## Fresh held-out sets (built before training, each evaluated ONCE)
- **Held-out families**: charity_relief + package_delivery exam
  rows (seeds_v9.json; 70 generated variants per family from ALL
  their templates; excluded from ALL v9 training and validation).
  v5 scored once as the baseline.
- **Obfuscation set D**: the seed-42 IMC25-test / UCI-test base
  pools are EXHAUSTED (v6 800/300 + B-v7 800/300 + set C 765/78 =
  all 2,365 test positives and all 678 test ham). Set D therefore
  draws NEW base rows from the VALIDATION pools, never used as
  obfuscation bases in any round: 800 IMC25 validation positives +
  300 UCI validation ham, seeded sample (shuffle seed
  "v9-obf-eval"). Transforms: the same never-trained set-B
  mechanisms at the B-v7/set-C parameter variants
  (gen_v7_data.OBF_B7), fresh per-row seeds (v9-obf-*). DISCLOSED:
  the clean forms of these rows sat in both lineages' validation
  sets (threshold selection only — neither v5 nor v9 trains on
  them); the obfuscated forms are new items; both models are
  scored once on identical rows. Legit FP on set D is reported,
  not gated (same as set C).
- **Trained-family internal splits (informational, NOT an exam)**:
  romance / grandparent / wrong_number_opener template-index-3
  rows, generated fresh for v9. Romance + grandparent recall
  reported BY NAME for v9 and for v5 on the same rows. (v8's exam
  rows for these families were generated from all templates with
  v8 seeds and are spent; v9's internal rows are fresh variants of
  the index-3 templates only.)
- **Frozen v5 suites** (modern, UCI, 2026-family, raw-form stress,
  transactional, probe) re-evaluated once as regression.

## Keeper gates (ALL must pass, fixed in advance)
1. Modern recall ≥ 0.965
2. Legit FPR ≤ 0.005 (modern test block)
3. Transactional legit FP ≤ 0.01
4. Obfuscation-D recall STRICTLY greater than v5's on set D
5. Held-out-family (charity_relief + package_delivery) recall
   ≥ v5's on the same families

If any gate fails: v9 is rejected, v5 stays the keeper,
`model_v9.npz` and `predict_v9.py` are NOT retained, and there is
no re-tuning against these spent tests.

## Budget
Colab cap for this run: 4 compute units (expect ≈0). CPU runtime
ONLY (std → highmem; NO GPU fallback — GPU jobs stay serialized
per the Colab CONNECTOR hard rules; CPU runs in parallel with GPU
jobs). Runtime deleted in a finally block; verify the v9 runtime is
gone afterwards and leave any GPU runtime untouched.

## Frontier-model contact — affirmation (v9)
Same as v1–v8: no Muse / Muse or OpenAI / Luna output is used as
training data, labels, scores, rankings, filtering, or selection.
All new text is deterministic code: seed templates are slot-filled
by augment_v4 / gen_v9_data from fixed lists. Training runs on
Colab as compute only. (Web sources above were read by the operator
to ground the template mechanics in the cited government
publications, the same standard as seeds_v4.json / seeds_v8.json.)

## Results (appended after the single run)

**Verdict: v9 is a KEEPER — all 5 gates pass. v9 replaces v5 as
the keeper.** First candidate to beat v5 in four rounds.
`model_v9.npz` (317,338 B, SHA-256 f24474c8…d3e8) is retained with
`predict_v9.py`; pure-NumPy runtime parity vs sklearn verified on
the 200-text validation sample (max probability difference
4.9e-07). Numbers in `metrics_v9.json`.

Run facts: Colab CPU runtime (std; no GPU fallback), created in
parallel with another job's T4 GPU runtime per the concurrency
rule — that runtime was verified untouched afterwards and the v9
runtime deleted in the finally block. Total driver time 217 s.
Balance 593.22 → 593.03 over the whole working window; the 0.19
delta is shared with the concurrent T4 job (1.15 units/hr) and is
fully explained by it — v9's own CPU share is ≈0.00–0.01 units,
far under the 4-unit cap. Two crashed attempts preceded the
successful run, both recorded here per the v6 precedent: (1) the
train script's leakage assert fired on set D before any training —
no-op transforms of validation-pool bases can reproduce a clean
validation text byte-for-byte, a construction bug in the assert's
assumptions, not data leakage into training; the fix was in
gen_v9_data (no-op transforms and exact collisions with any clean
train/validation text are excluded during the pool walk), verified
locally against the rebuilt key sets before re-running. (2) The
bundle was missing `predict_v4.py` (imported by probe_legit); the
script trained and computed its evaluations in memory but crashed
before printing or saving anything — no value from that attempt
was observed or used in any decision, and the recipe was re-run
unchanged. Data-construction fixes before the run, also recorded:
wno-2 gained {NAME}/{DAY} slots (it was slot-free and yielded only
13 unique validation variants), and the conversational-legit
templates gained wider slot lists (legit group landed at 1,083
train / 285 validation rows vs the ≈1,500/≈300 target; fam9
validation 150 rows).

Threshold: v5's rule was satisfied outright — chosen **0.60**
(validation legit FPR 0.0096 ≤ 0.010, validation legit n=1,256,
validation scam recall 0.9366). No fallback needed. This is the
mechanism the cap rounds missed: adding conversational legit data
to VALIDATION changed the legit distribution the rule sees, so
the rule itself chose a much lower threshold than v5's 0.77 — and
recall rose on every frozen suite while test FPR stayed in gate.

| Gate | v9 | v5 | Pass |
|---|---|---|---|
| Modern recall ≥ 0.965 | 0.9784 | 0.9674 | ✓ |
| Legit FPR ≤ 0.005 | 0.0044 | 0.0029 | ✓ |
| Transactional FP ≤ 0.01 | 0.0035 | 0.0035 | ✓ |
| Obfuscation-D recall > v5 | 0.9350 | 0.9200 | ✓ |
| Held-out-family recall ≥ v5 | 0.9429 | 0.9000 | ✓ |

Frozen suites (each evaluated once, threshold 0.60): UCI recall
**0.9792** (v5 0.9583 — best of any version); 2026-family 0.9926
(v5 0.9758), with every family at 1.000 except v4's wrong_number
at 0.6667 (8/12, up from 0.000 in v4–v8); raw-form stress 0.9564
(v5 0.9463); transactional scam recall 1.0000 with legit FP
identical to v5 (1/285). Probe: mean prob 0.0872 (v5 0.1053),
2/34 ≥ threshold (v5 1/34), 1/34 ≥ 0.70 (same as v5) — marginally
more probe pressure at the lower threshold, still low in absolute
terms. Obfuscation set D legit FP: v9 0.0100 vs v5 0.0067
(reported, not gated — the one legit-side number where v9 is
worse; bases are validation-pool ham, disclosed in the design).

Trained-family internal splits (template index 3, informational):
**romance 0.9714 (v5: 0.0143), grandparent 0.9857 (v5: 0.0143)** —
the conversational/emotional register that scored 0.14–0.21 for
both models on v8's exam is now captured where sibling templates
were trained. Fresh exam: charity_relief 0.8857 (v5 0.8000),
package_delivery 1.0000 (v5 1.0000).

**The one clear failure: wrong_number_opener internal test 0.0000
for BOTH v9 and v5 (n=70).** Training on five sibling opener
templates did not transfer to the held-out opener template at
all — its opening moves ("Oops, I think I have the wrong number.
Sorry to bother you!") now sit in a register the model has also
been taught is legitimate chat, and the pivot ("my friend taught
me a way to make $X a month from my phone") does not outweigh
them. The soft-opener problem survives v9 exactly as it survived
v4; flagging it harder would likely cost real false alarms on
genuine wrong-number texts, which this audience sends and
receives often. Recorded as the standing known gap, not tuned
against.

Standing conclusion after v9: the winning lever was DATA on both
sides of the conversational register, fed through v5's untouched
recipe and threshold rule — not feature widening (v6), threshold
caps (v7/v8), or added groups of the old kind. v9 is the keeper;
v5's files are retained for comparison. Remaining known gaps:
wrong_number_opener (above), password-changed/security notices
(v5's probe caveat, unchanged), and all the standing v5
limitations on real-world data (masked IMC25 optimism, template-
based transactional coverage, no consented real senior corpus).
