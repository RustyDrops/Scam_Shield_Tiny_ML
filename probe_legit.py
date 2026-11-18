#!/usr/bin/env python3
"""EVAL-ONLY probe: modern legitimate transactional messages scored by a
Scam Shield model. These hand-written probes are NEVER training data —
they exist to expose the v4 false-positive flaw on message types that
barely exist in 2011-era UCI ham (2FA codes, appointment reminders,
pharmacy/school/tracking/bank alerts)."""
import sys
import predict_v4

PROBES = [
    # verification codes, inform-only (real services' phrasing)
    "Your verification code is 482916. Don't share it with anyone.",
    "Your Google code is 731204. Never share this code with anyone.",
    "482916 is your Microsoft account verification code.",
    "Your Chase verification code is 663120. We will never ask for this code.",
    "Your one-time passcode is 902184. It expires in 10 minutes. Do not share it.",
    "Apple ID code: 551207. Do not share it with anyone.",
    "Your Venmo code is 308475. Venmo will never call you to ask for this code.",
    "Your login code is 7719. If you didn't request this, you can ignore this message.",
    "Steam Guard code: QK7TP. This code expires in 5 minutes.",
    "Your two-factor authentication code is 284 619.",
    # appointment reminders
    "Reminder: You have a dentist appointment tomorrow at 2:30 PM with Dr. Patel at Bright Smiles Dental. Reply C to confirm or call 555-0134 to reschedule.",
    "Reminder from Ada Family Clinic: Samuel has an appointment on Thursday at 9:15 AM. Please arrive 10 minutes early.",
    "Your hair appointment at Main Street Salon is confirmed for Saturday at 11 AM.",
    "Reminder: Your eye exam at Vision Center is tomorrow at 3:45 PM. Bring your insurance card.",
    "Ada Veterinary Clinic: Bella's checkup is scheduled for Monday at 10:30 AM.",
    # pharmacy
    "CVS Pharmacy: Your prescription is ready for pickup at the Ada store. It will be held until Friday.",
    "Walgreens: Your refill for lisinopril is ready. Pickup by 8 PM today.",
    "Your prescription order has shipped and will arrive Thursday.",
    # school
    "Ada Public Schools: School is closed tomorrow, Jan 15, due to icy roads. Classes resume Friday.",
    "Reminder: Parent-teacher conferences are Thursday evening. Sign up in the school office.",
    "School bus 12 is running about 15 minutes late this morning.",
    # package tracking, no payment ask
    "USPS: Your package is out for delivery today. Tracking 9400 1000 0000 0000 0000 00.",
    "FedEx: Your package was delivered at 1:42 PM and left at the front door.",
    "UPS: Your shipment is on the way and scheduled for delivery tomorrow by 7 PM.",
    "Amazon: Your order of paper towels has shipped. Arriving Tuesday.",
    # bank / account alerts, no link, no ask
    "Chase alert: A deposit of $1,240.00 was posted to your account ending in 4821.",
    "Your checking account balance is $2,318.44 as of this morning.",
    "Wells Fargo: Your debit card ending in 0093 was used for $42.17 at Walmart.",
    "Your password was changed successfully. If this wasn't you, call the number on the back of your card.",
    "New sign-in to your account from a device in Oklahoma. If this wasn't you, open the app to review.",
    # misc transactional
    "Your library book is due in 2 days. Renew at the front desk or in the library app.",
    "OG&E: Your bill of $86.42 is ready. Your autopay will run on the 21st.",
    "Your table for 2 at Olive Garden is confirmed for Friday at 6:30 PM.",
    "Southwest: Check in for flight 2841 to Dallas opens 24 hours before departure.",
]

def main():
    model = predict_v4
    if len(sys.argv) > 1 and sys.argv[1] == "v5":
        import predict_v5
        model = predict_v5
    thr = model.THR
    scores = [(t, model.prob_scam(t)) for t in PROBES]
    ge_thr = sum(1 for _, p in scores if p >= thr)
    ge_70 = sum(1 for _, p in scores if p >= 0.70)
    ge_30 = sum(1 for _, p in scores if p >= 0.30)
    print(f"model threshold: {thr}")
    print(f"probes: {len(scores)} legit transactional")
    print(f"scored >= threshold ({thr}): {ge_thr}")
    print(f"scored >= 0.70: {ge_70}")
    print(f"scored >= 0.30: {ge_30}")
    for t, p in sorted(scores, key=lambda s: -s[1]):
        if p >= 0.30:
            print(f"  {p:.3f}  {t[:90]}")

if __name__ == "__main__":
    main()
