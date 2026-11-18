#!/usr/bin/env python3
"""Generate the v5 transactional dataset (deterministic code only).

Purpose: fix v4's false-positive flaw on modern legitimate transactional
texts (2FA codes, appointment/pharmacy/school reminders, tracking,
bank alerts). A hand-written probe of 34 such texts scored 26/34 >= 0.70
under v4 (probe_legit.py — eval only, never training data).

Method: programmatic templates with slot-filling, written for this fix.
- LEGIT templates: transactional messages that INFORM only (a code with
  "don't share it", a reminder, a tracking notice, a balance alert) —
  no link, no payment ask, no reply-with-secrets ask.
- SCAM templates: the counterpart messages whose distinguishing move is
  the ASK — send/share/reply with the code, enter it at a link, pay a
  fee to release/confirm — so the model can learn ask-vs-inform instead
  of code-vs-nocode.
Split is BY TEMPLATE within each category (index 1 -> validation,
index 3 -> test, rest -> train), same discipline as v4.

Outputs under data/v5/: txn_train.jsonl / txn_validation.jsonl /
txn_test.jsonl with {text, label, category, template_id}; manifest.json.
"""
import json
import os
import random
import re

import augment_v4

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "v5")
VARIANTS = {"train": 70, "validation": 45, "test": 55}

BRANDS_CODE = ["Google", "Microsoft", "Apple", "Chase", "Venmo", "PayPal",
               "Amazon", "Facebook", "Instagram", "Steam", "Netflix",
               "Wells Fargo", "Bank of America", "your bank", "Cash App",
               "WhatsApp", "Uber", "DoorDash", "AT&T", "Verizon"]
CLINICS = ["Bright Smiles Dental", "Ada Family Clinic", "Vision Center",
           "Main Street Dental", "Lakeside Pediatrics", "City Eye Care",
           "Ada Veterinary Clinic", "Downtown Hair Studio",
           "Main Street Salon", "QuickCare Clinic", "Family Eye Center"]
PROVIDERS = ["Dr. Patel", "Dr. Nguyen", "Dr. Smith", "Dr. Garcia",
             "Dr. Johnson", "Dr. Lee", "Dr. Brown"]
PHARMACIES = ["CVS Pharmacy", "Walgreens", "Walmart Pharmacy",
              "Ada Drug", "Rite Aid", "Main Street Pharmacy"]
MEDS = ["lisinopril", "metformin", "atorvastatin", "amoxicillin",
        "your blood pressure medication", "your inhaler refill",
        "levothyroxine", "omeprazole"]
SCHOOLS = ["Ada Public Schools", "Lincoln Elementary", "Washington Middle School",
           "Ada High School", "Jefferson Elementary"]
COURIERS = ["USPS", "FedEx", "UPS", "Amazon", "DHL"]
MERCHANTS = ["Walmart", "Target", "Amazon", "Shell", "Dollar General",
             "McDonald's", "AT&T", "Netflix"]
ITEMS = ["paper towels", "dog food", "a phone charger", "laundry detergent",
         "coffee", "batteries", "shampoo", "vitamins"]
PETS = ["Bella", "Max", "Luna", "Charlie", "Daisy", "Rocky"]
BANKS = ["Chase", "Wells Fargo", "Bank of America", "US Bank",
         "your credit union", "Capital One"]

# ---------------- templates: (category, label, [templates]) ----------------
T = []
def cat(category, label, templates):
    T.append((category, label, templates))

cat("code_inform", 0, [
    "Your verification code is {CODE}. Don't share it with anyone.",
    "Your {BRAND} code is {CODE}. Never share this code with anyone.",
    "{CODE} is your {BRAND} verification code.",
    "Your {BRAND} verification code is {CODE}. We will never ask for this code.",
    "Your one-time passcode is {CODE}. It expires in {N} minutes. Do not share it.",
    "{BRAND} code: {CODE}. Do not share it with anyone.",
    "Your login code is {CODE}. If you didn't request this, you can ignore this message.",
    "Your two-factor authentication code is {CODES}. It expires in {N} minutes.",
    "Use {CODE} to finish signing in to {BRAND}. This code expires soon.",
    "{BRAND}: {CODE} is your security code. Never share your code - we will never call and ask for it.",
])
cat("appointment", 0, [
    "Reminder: You have a dentist appointment tomorrow at {TIME} with {PROVIDER} at {CLINIC}. Reply C to confirm or call {PHONE} to reschedule.",
    "Reminder from {CLINIC}: {NAME} has an appointment on {DAY} at {TIME}. Please arrive 10 minutes early.",
    "Your appointment at {CLINIC} is confirmed for {DAY} at {TIME}.",
    "{CLINIC}: {PET}'s checkup is scheduled for {DAY} at {TIME}.",
    "Reminder: Your appointment at {CLINIC} is tomorrow at {TIME}. Bring your insurance card.",
    "{CLINIC} reminder: See you {DAY} at {TIME} for your appointment. Call {PHONE} if you need to reschedule.",
])
cat("pharmacy", 0, [
    "{PHARMACY}: Your prescription is ready for pickup. It will be held until {DAY}.",
    "{PHARMACY}: Your refill for {MED} is ready. Pickup by {TIME} today.",
    "Your prescription order has shipped and will arrive {DAY}.",
    "{PHARMACY}: Your prescription for {MED} is ready for pickup at the drive-thru.",
    "{PHARMACY} reminder: You have {N} prescriptions ready for pickup.",
])
cat("school", 0, [
    "{SCHOOL}: School is closed tomorrow due to bad weather. Classes resume {DAY}.",
    "{SCHOOL}: Reminder: Parent-teacher conferences are {DAY} evening.",
    "{SCHOOL}: Bus {N2} is running about {N} minutes late this morning.",
    "{SCHOOL}: Early dismissal today at {TIME} for teacher training.",
    "{SCHOOL}: Report cards are now available in the parent portal app.",
])
cat("tracking", 0, [
    "{COURIER}: Your package is out for delivery today. Tracking {TRACK}.",
    "{COURIER}: Your package was delivered at {TIME} and left at the front door.",
    "{COURIER}: Your shipment is on the way and scheduled for delivery {DAY} by {TIME}.",
    "Amazon: Your order of {ITEM} has shipped. Arriving {DAY}.",
    "{COURIER}: A package is arriving today by {TIME}.",
    "Your order of {ITEM} is out for delivery.",
])
cat("bank_alert", 0, [
    "{BANK} alert: A deposit of ${AMT} was posted to your account ending in {LAST4}.",
    "Your checking account balance is ${AMT} as of this morning.",
    "{BANK}: Your debit card ending in {LAST4} was used for ${FEE} at {MERCHANT}.",
    "Your password was changed successfully. If this wasn't you, call the number on the back of your card.",
    "New sign-in to your account from a device in {STATE}. If this wasn't you, open the app to review.",
    "{BANK}: Your statement for account ending in {LAST4} is ready in the app.",
    "{BANK} alert: Your autopay of ${FEE} to {MERCHANT} is scheduled for {DAY}.",
])
cat("misc_txn", 0, [
    "Your library book is due in {N2} days. Renew at the front desk or in the library app.",
    "Your electric bill of ${FEE} is ready. Your autopay will run on the {N2}th.",
    "Your table for {N2} at {RESTAURANT} is confirmed for {DAY} at {TIME}.",
    "Your flight check-in is now open. Flight {N3} departs {DAY} at {TIME}.",
    "Your hotel check-in is {DAY} from {TIME}. See you soon.",
])
cat("code_ask", 1, [
    "We just texted you a verification code. Send us the code to confirm your identity and complete your refund.",
    "Reply with the {BRAND} verification code you just received so we can verify your account now.",
    "To prove you're a real seller, text me the {N} digit code you receive. I need it before I send payment.",
    "Share your verification code with our agent on this call to stop the cancellation of your account.",
    "Enter the code we sent you at {URL} to claim your refund.",
    "Your account will be locked. Send back the security code we texted you to keep it open.",
    "This is {BRAND} support. Give the {N}-digit code from your messages to verify you are the account owner.",
    "We sent a code to your phone. Reply with it here to unlock your earnings.",
    "Forward the verification code you just got to this number to finish your registration.",
])
cat("appointment_phish", 1, [
    "Your appointment at {CLINIC} will be cancelled. Pay the ${FEE} no-show fee now to keep it: {URL}",
    "{CLINIC}: Confirm your appointment and update your insurance details today at {URL} or lose your slot.",
    "You missed your appointment. Pay the ${FEE} missed-appointment fee here: {URL}",
    "{CLINIC}: Your appointment is on hold. Verify your payment information at {URL} to keep your booking.",
])
cat("pharmacy_phish", 1, [
    "{PHARMACY}: Your refill is on hold. Update your payment information at {URL} to release it.",
    "Your prescription cannot be filled. Pay the ${FEE} processing fee at {URL} to continue.",
    "{PHARMACY} alert: Action required. Confirm your card details at {URL} or your prescription will be returned.",
])
cat("tracking_phish", 1, [
    "{COURIER}: Your package could not be delivered. Pay the ${FEE} redelivery fee at {URL}",
    "{COURIER}: Customs fee of ${FEE} due on your parcel. Pay now to release it: {URL}",
    "Your package is held at the warehouse. Confirm your address and card at {URL} within 24 hours.",
])
cat("bank_alert_phish", 1, [
    "{BANK}: Unusual activity detected. Verify your account now at {URL} or it will be suspended.",
    "Your account has been locked. Enter your code and password at {URL} to restore access.",
    "{BANK} alert: Reply with your PIN and the code we sent to stop this transaction.",
    "A payment of ${AMT} is pending. If this wasn't you, confirm your identity at {URL}",
    "{BANK}: Your card ending in {LAST4} is blocked. Call {PHONE} and read out the verification code we text you to unblock it.",
])

RESTAURANTS = ["Olive Garden", "Chili's", "Applebee's", "Red Lobster",
               "Texas Roadhouse"]

def fill(text, rng):
    def code(m):
        n = rng.choice([4, 6, 6, 6, 8])
        return "".join(rng.choice("0123456789") for _ in range(n))
    text = re.sub(r"\{CODE\}", code, text)
    text = text.replace("{CODES}", f"{rng.randint(100,999)} {rng.randint(100,999)}")
    text = text.replace("{BRAND}", rng.choice(BRANDS_CODE))
    text = text.replace("{BANK}", rng.choice(BANKS))
    text = text.replace("{CLINIC}", rng.choice(CLINICS))
    text = text.replace("{PROVIDER}", rng.choice(PROVIDERS))
    text = text.replace("{PHARMACY}", rng.choice(PHARMACIES))
    text = text.replace("{MED}", rng.choice(MEDS))
    text = text.replace("{SCHOOL}", rng.choice(SCHOOLS))
    text = text.replace("{COURIER}", rng.choice(COURIERS))
    text = text.replace("{MERCHANT}", rng.choice(MERCHANTS))
    text = text.replace("{ITEM}", rng.choice(ITEMS))
    text = text.replace("{PET}", rng.choice(PETS))
    text = text.replace("{RESTAURANT}", rng.choice(RESTAURANTS))
    text = text.replace("{TIME}", f"{rng.randint(1,12)}:{rng.choice(['00','15','30','45'])} {rng.choice(['AM','PM'])}")
    text = text.replace("{LAST4}", f"{rng.randint(1000,9999)}")
    text = text.replace("{TRACK}", f"{rng.randint(1000,9999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)}")
    text = text.replace("{N3}", f"{rng.randint(100,9999)}")
    text = text.replace("{N2}", str(rng.randint(2, 28)))
    text = text.replace("{N}", str(rng.choice([5, 6, 10, 15, 20, 30])))
    text = text.replace("{STATE}", rng.choice(augment_v4.STATES))
    # shared slots handled by v4 code: {AMT} {FEE} {DAY} {PHONE} {URL} {NAME}
    text = augment_v4.fill_slots(text, rng, "generic")
    return text

def split_of(idx):
    if idx == 1:
        return "validation"
    if idx == 3:
        return "test"
    return "train"

def main():
    os.makedirs(OUT, exist_ok=True)
    handles = {s: open(os.path.join(OUT, f"txn_{s}.jsonl"), "w",
                       encoding="utf-8")
               for s in ("train", "validation", "test")}
    counts = {}
    n_templates = {}
    for category, label, templates in T:
        for idx, tpl in enumerate(templates):
            split = split_of(idx)
            n_templates[split] = n_templates.get(split, 0) + 1
            rng = random.Random(f"v5-{split}-{category}-{idx}")
            seen = set()
            for _ in range(VARIANTS[split]):
                txt = fill(tpl, rng)
                if txt in seen:
                    continue
                seen.add(txt)
                handles[split].write(json.dumps(
                    {"text": txt, "label": label, "category": category,
                     "template_id": f"{category}-{idx}"}) + "\n")
                key = (split, label)
                counts[key] = counts.get(key, 0) + 1
    for h in handles.values():
        h.close()
    manifest = {
        "rows_by_split_label": {f"{s}/{'scam' if l else 'legit'}": n
                                for (s, l), n in sorted(counts.items())},
        "templates_per_split": n_templates,
        "variants_per_template": VARIANTS,
        "split_rule": "per category: template index 1 -> validation, "
                      "index 3 -> test, others -> train",
        "generation": "deterministic slot-filling code only (gen_v5_data.py); "
                      "no LLM text; probe_legit.py probes are eval-only and "
                      "are NOT in this data",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
