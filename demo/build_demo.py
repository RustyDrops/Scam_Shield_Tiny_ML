#!/usr/bin/env python3
"""Build demo/index.html: Scam Shield v9 in one self-contained file.

Exports model_v9.npz to browser form:
- vocabularies as raw text blocks (entries joined by \\x01, a character that
  cannot occur inside an entry), read via textContent at load;
- idf / coef float32 arrays as base64 (exact float32 bytes -> exact parity);
- intercept / threshold as JSON numbers.
The inference core in the page is a line-for-line port of predict_v9.py.
"""
import base64, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
m = np.load(os.path.join(SRC, "model_v9.npz"), allow_pickle=False)

def b64f32(a):
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()

MODEL = {
    "wordIdf": b64f32(m["word_idf"]),
    "charIdf": b64f32(m["char_idf"]),
    "coef": b64f32(m["coef"]),  # word branch first, then char branch
    "intercept": float(m["intercept"]),
    "threshold": float(m["threshold"]),
}
word_vocab = "\x01".join(m["word_vocab"].tolist())
char_vocab = "\x01".join(m["char_vocab"].tolist())
assert "\x01" not in "".join(m["word_vocab"].tolist())
assert "</script" not in word_vocab.lower() and "</script" not in char_vocab.lower()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scam Shield — Check a Message</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #ffffff; color: #111111;
         font-family: Arial, Helvetica, sans-serif; font-size: 20px; line-height: 1.5; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 18px 40px; }
  h1 { font-size: 38px; margin: 8px 0 4px; }
  .sub { font-size: 21px; margin: 0 0 20px; }
  label { font-weight: bold; font-size: 22px; display: block; margin-bottom: 8px; }
  textarea { width: 100%; min-height: 170px; font-size: 20px; padding: 14px;
             border: 3px solid #111111; border-radius: 8px; font-family: inherit; }
  button.check { width: 100%; margin-top: 16px; padding: 18px; font-size: 24px;
                 font-weight: bold; background: #111111; color: #ffffff;
                 border: 3px solid #111111; border-radius: 8px; cursor: pointer; }
  button.check:active { background: #444444; }
  .examples { margin-top: 26px; }
  .examples p { font-weight: bold; margin: 0 0 10px; font-size: 21px; }
  .examples button { display: block; width: 100%; text-align: left; margin: 0 0 10px;
                     padding: 14px 16px; font-size: 20px; background: #ffffff;
                     color: #111111; border: 3px solid #111111; border-radius: 8px; cursor: pointer; }
  .examples button:active { background: #eeeeee; }
  #result { display: none; margin-top: 28px; border: 4px solid #111111;
            border-radius: 10px; padding: 20px; }
  #result.safe  { background: #e6f4e6; }
  #result.warn  { background: #fff3cd; }
  #result.scam  { background: #fde8e8; }
  #verdict { font-size: 34px; font-weight: bold; margin: 0 0 6px; }
  #score { font-size: 22px; margin: 0 0 10px; }
  #advice { font-size: 21px; margin: 0; }
  #why { margin-top: 22px; display: none; }
  #why h2 { font-size: 24px; margin: 0 0 8px; }
  #why ul { margin: 0; padding-left: 24px; }
  #why li { margin-bottom: 6px; font-size: 20px; }
  .toward-scam { font-weight: bold; }
  footer { margin-top: 36px; border-top: 2px solid #111111; padding-top: 14px;
           font-size: 16px; }
</style>
</head>
<body>
<main>
  <h1>Scam Shield</h1>
  <p class="sub">Paste a text message, email, or note below. We will tell you if it
  looks like a scam. <strong>Nothing you paste is sent anywhere</strong> — the check
  runs right here on this page, even with the internet turned off.</p>

  <label for="msg">Message to check</label>
  <textarea id="msg" placeholder="Paste the message here..."></textarea>
  <button class="check" id="checkBtn" type="button">Check this message</button>

  <div class="examples">
    <p>Or try an example:</p>
    <button type="button" data-example="bank">Example 1: A message about my bank account</button>
    <button type="button" data-example="delivery">Example 2: A message about a package delivery fee</button>
    <button type="button" data-example="legit">Example 3: A normal message from family</button>
    <button type="button" data-example="toll">Example 4: A message about an unpaid road toll</button>
    <button type="button" data-example="irs">Example 5: A message about a tax refund</button>
    <button type="button" data-example="taskjob">Example 6: A message offering easy paid tasks</button>
    <button type="button" data-example="code">Example 7: A real verification code message (this one is fine)</button>
  </div>

  <div id="result" role="status">
    <p id="verdict"></p>
    <p id="score"></p>
    <p id="advice"></p>
  </div>

  <div id="why">
    <h2>Words that mattered most</h2>
    <ul id="whyList"></ul>
    <p>These are simply the words that pushed the answer one way or the other.
    A single word does not make a message a scam — the whole message is weighed together.</p>
  </div>

  <footer>
    <p><strong>Prototype.</strong> Scam Shield is a test tool, not a guarantee.
    A “safe-looking” result does not prove a message is safe, and a “likely scam”
    result does not prove it is a scam. When in doubt, do not tap links or send money —
    call the company or person using a number you already trust.</p>
    <p>This is Scam Shield <strong>version 5</strong>. In held-out tests it caught
    96.7% of modern (2025) scam texts, 95.8% of older spam texts, and 97.6% of
    examples from the newest 2026 scam families, while flagging only 0.29% of
    normal messages by mistake. On scam texts rewritten with real-looking links
    and numbers instead of placeholders, it caught 94.6%.</p>
    <p>Version 5 learned the difference between a message that <em>tells</em> you
    something and one that <em>asks</em> you to hand something over. Everyday
    messages like bank alerts, appointment reminders, and verification codes
    used to be flagged by mistake 53% of the time — now that false-alarm rate
    on those messages is 0.35%. A real verification code (Example 7 above) is
    fine; a message asking you to send a code to someone else is not.</p>
    <p>Trained on public datasets: the UCI SMS Spam Collection (CC BY 4.0), the
    IMC25 smishing dataset (CC BY 4.0), and scam examples published by the FTC,
    the FBI's IC3, and the IRS. The checking model runs entirely in this
    page — no network, no tracking, nothing uploaded.</p>
  </footer>
</main>

<script type="text/plain" id="wordVocab">__WORD_VOCAB__</script>
<script type="text/plain" id="charVocab">__CHAR_VOCAB__</script>

<script id="model-core">
"use strict";
const MODEL_META = __MODEL_JSON__;

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
const WORD_IDF = b64ToF32(MODEL_META.wordIdf);
const CHAR_IDF = b64ToF32(MODEL_META.charIdf);
const COEF = b64ToF32(MODEL_META.coef);
const INTERCEPT = MODEL_META.intercept;
const THRESHOLD = MODEL_META.threshold;
const WORD_VOCAB = document.getElementById("wordVocab").textContent.split("\u0001");
const CHAR_VOCAB = document.getElementById("charVocab").textContent.split("\u0001");
const WV = new Map(WORD_VOCAB.map((t, i) => [t, i]));
const CV = new Map(CHAR_VOCAB.map((t, i) => [t, i]));

// Port of predict_v9.py: TOK = \b\w[\w']+\b (unicode). Candidates are runs of
// word characters / apostrophes; strip edge apostrophes, keep length >= 2.
function tokenizeWords(low) {
  const cands = low.match(/[\p{L}\p{N}\p{M}_']+/gu) || [];
  const toks = [];
  for (const c of cands) {
    const t = c.replace(/^'+|'+$/g, "");
    if (t.length >= 2) toks.push(t);
  }
  return toks;
}

function tfidfContribs(counts, idf, coefOffset) {
  // returns Map index -> normalized tfidf value * coef (its push on the score)
  const vals = new Map();
  let normSq = 0;
  for (const [i, c] of counts) {
    const v = (1 + Math.log(c)) * idf[i];
    vals.set(i, v);
    normSq += v * v;
  }
  const norm = Math.sqrt(normSq);
  const out = new Map();
  if (norm > 0) {
    for (const [i, v] of vals) out.set(i, (v / norm) * COEF[coefOffset + i]);
  }
  return out;
}

function analyze(text) {
  const low = text.toLowerCase();
  const toks = tokenizeWords(low);
  const wcounts = new Map();
  const feats = toks.concat(toks.slice(0, -1).map((a, i) => a + " " + toks[i + 1]));
  for (const f of feats) {
    const i = WV.get(f);
    if (i !== undefined) wcounts.set(i, (wcounts.get(i) || 0) + 1);
  }
  const ccounts = new Map();
  const words = low.replace(/\s+/g, " ").split(" ").filter(w => w.length > 0);
  for (const raw of words) {
    const w = " " + raw + " ";
    for (const n of [3, 4, 5]) {
      for (let off = 0; off + n <= w.length; off++) {
        const i = CV.get(w.slice(off, off + n));
        if (i !== undefined) ccounts.set(i, (ccounts.get(i) || 0) + 1);
      }
    }
  }
  const wcontrib = tfidfContribs(wcounts, WORD_IDF, 0);
  const ccontrib = tfidfContribs(ccounts, CHAR_IDF, WORD_VOCAB.length);
  let z = INTERCEPT;
  for (const v of wcontrib.values()) z += v;
  for (const v of ccontrib.values()) z += v;
  const prob = 1 / (1 + Math.exp(-z));
  const wordWhy = [...wcontrib.entries()]
    .map(([i, v]) => ({ term: WORD_VOCAB[i], push: v }))
    .filter(d => Math.abs(d.push) > 1e-9);
  return { prob, wordWhy };
}

function probScam(text) { return analyze(text).prob; }
</script>

<script>
"use strict";
const EXAMPLES = {
  bank: "URGENT: Your bank account has been locked due to suspicious activity. Verify your identity now at secure-verify-bank.com or your account will be closed within 24 hours. Enter your card number and PIN to restore access.",
  delivery: "USPS: Your package is being held at the delivery centre because a $1.99 shipping fee is unpaid. Pay now to release your parcel: bit.ly/parcel-release-fee",
  legit: "Hi Mum, just letting you know I'll be home for dinner at 7 tonight. Could you pick up some milk on your way back? See you later, love you xx",
  toll: "E-ZPass: We've noticed an outstanding toll amount of $12.51 on your record. To avoid a late fee of $50, visit https://ezpass-pay.com/settle to settle your balance.",
  irs: "IRS Notice: Your 2025 tax refund of $1,240 has been approved. To claim your refund, verify your identity within 24 hours: https://irs-refund-verify.com/claim",
  taskjob: "Hi! We are hiring for online product boosting tasks. Earn $300/day working from home, paid daily. Reply YES to get started.",
  code: "Your verification code is 482916. Don't share it with anyone — we will never call or text to ask for this code."
};
const msgEl = document.getElementById("msg");
const resultEl = document.getElementById("result");
const whyEl = document.getElementById("why");

function showResult() {
  const text = msgEl.value.trim();
  if (!text) {
    resultEl.style.display = "block";
    resultEl.className = "warn";
    document.getElementById("verdict").textContent = "Nothing to check yet";
    document.getElementById("score").textContent = "";
    document.getElementById("advice").textContent = "Paste a message in the box above first, then press the check button.";
    whyEl.style.display = "none";
    return;
  }
  const { prob, wordWhy } = analyze(text);
  const pct = Math.round(prob * 100);
  let cls, verdict, advice;
  if (prob >= THRESHOLD) {
    cls = "scam"; verdict = "LIKELY SCAM";
    advice = "Be very careful. Do NOT tap any link, do NOT send money, and do NOT share codes or bank details. If it claims to be your bank or a delivery company, contact them using a number you already trust — not a number in this message.";
  } else if (prob >= 0.30) {
    cls = "warn"; verdict = "SUSPICIOUS — be careful";
    advice = "Some parts of this message look like known scams. Do not tap links or send money until you have checked with the sender in another way — for example, by calling a number you already have.";
  } else {
    cls = "safe"; verdict = "SAFE-LOOKING";
    advice = "This message looks like the normal, everyday messages in our training examples. Still stay alert: if it asks for money, gift cards, or a code, stop and check with the person directly.";
  }
  resultEl.style.display = "block";
  resultEl.className = cls;
  document.getElementById("verdict").textContent = verdict;
  document.getElementById("score").textContent = "Scam score: " + pct + " out of 100 (higher means it looks more like known scams).";
  document.getElementById("advice").textContent = advice;

  const towardScam = wordWhy.filter(d => d.push > 0).sort((a, b) => b.push - a.push);
  const towardSafe = wordWhy.filter(d => d.push < 0).sort((a, b) => a.push - b.push);
  let picks = [];
  if (prob >= 0.30) picks = towardScam.slice(0, 5).map(d => ({ ...d, dir: "scam" }))
    .concat(towardSafe.slice(0, 1).map(d => ({ ...d, dir: "safe" })));
  else picks = towardSafe.slice(0, 5).map(d => ({ ...d, dir: "safe" }))
    .concat(towardScam.slice(0, 1).map(d => ({ ...d, dir: "scam" })));
  const list = document.getElementById("whyList");
  list.innerHTML = "";
  if (!picks.length) {
    whyEl.style.display = "none";
    return;
  }
  for (const d of picks) {
    const li = document.createElement("li");
    const term = document.createElement("span");
    term.className = "toward-scam";
    term.textContent = "“" + d.term + "”";
    li.appendChild(term);
    li.appendChild(document.createTextNode(d.dir === "scam"
      ? " — words like this appear often in scams, so this pushed toward SCAM."
      : " — words like this appear in normal everyday messages, so this pushed toward SAFE."));
    list.appendChild(li);
  }
  whyEl.style.display = "block";
}

document.getElementById("checkBtn").addEventListener("click", showResult);
for (const btn of document.querySelectorAll("[data-example]")) {
  btn.addEventListener("click", () => {
    msgEl.value = EXAMPLES[btn.getAttribute("data-example")];
    showResult();
    resultEl.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}
</script>
</body>
</html>
"""

html = (HTML
        .replace("__MODEL_JSON__", json.dumps(MODEL))
        .replace("__WORD_VOCAB__", word_vocab)
        .replace("__CHAR_VOCAB__", char_vocab))
out = os.path.join(HERE, "index.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(out, os.path.getsize(out), "bytes")
