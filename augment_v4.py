#!/usr/bin/env python3
"""Deterministic augmentation for Scam Shield v4.

Two jobs, both pure code with seeded RNGs (no LLM text anywhere):
1. Slot-filling for the government-sourced seed templates in seeds_v4.json:
   amounts, dates, agency/brand/state names come from fixed lists below;
   phone numbers are fictional 555 numbers; URLs are lookalike domains
   assembled by code patterns. A few mechanical surface transforms
   (prefix labels, trailing punctuation) add variety.
2. De-masking of IMC25 rows: replaces the researchers' placeholders
   (<URL>, <PHONE_NUMBER>, <DATE_TIME>, <NAMED_ENTITY>, ...) with
   code-generated raw forms, producing the raw-form variant used to fix
   v3's known masked-placeholder weakness.
"""
import random
import re

# ---------------- fixed slot lists ----------------
STATES = ["Alabama", "Arizona", "Arkansas", "California", "Colorado",
          "Connecticut", "Florida", "Georgia", "Illinois", "Indiana",
          "Iowa", "Kansas", "Kentucky", "Louisiana", "Maryland",
          "Massachusetts", "Michigan", "Minnesota", "Missouri", "Nevada",
          "New Jersey", "New Mexico", "New York", "North Carolina", "Ohio",
          "Oklahoma", "Oregon", "Pennsylvania", "South Carolina",
          "Tennessee", "Texas", "Utah", "Virginia", "Washington",
          "Wisconsin"]
FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer",
               "Michael", "Linda", "David", "Elizabeth", "William", "Barbara",
               "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
               "Charles", "Karen", "Daniel", "Nancy", "Matthew", "Lisa",
               "Anthony", "Betty", "Mark", "Margaret", "Donald", "Sandra"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
              "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee"]
TOLL_AGENCIES = ["E-ZPass", "SunPass", "TxTag", "Peach Pass", "FasTrak",
                 "ExpressToll", "PlatePay", "State Toll Service",
                 "Toll Road Service", "Turnpike Toll Services"]
COUNTIES = ["Jefferson County", "Lincoln County", "Madison County",
            "Franklin County", "Washington County", "Greene County",
            "Monroe County", "Salem County", "Marion County", "Warren County"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
AREACODES = ["800", "888", "877", "866", "855", "202", "312", "415", "646",
             "713", "214", "602"]
ORGS = ["Chase Bank", "Bank of America", "Wells Fargo", "Citibank",
        "Amazon", "USPS", "FedEx", "UPS", "AT&T", "Verizon", "Walmart",
        "the IRS", "Social Security Administration", "Apple", "Microsoft",
        "PayPal", "Venmo", "Cash App", "your bank", "your card issuer"]
CITIES = ["Dallas, TX", "Houston, TX", "Phoenix, AZ", "Atlanta, GA",
          "Chicago, IL", "Denver, CO", "Miami, FL", "Seattle, WA",
          "Columbus, OH", "Charlotte, NC", "Portland, OR", "Las Vegas, NV"]

URL_CORES = {
    "toll": ["ezpass", "sunpass", "txtag", "tollpay", "turnpiketollservices",
             "tollservice", "paytoll"],
    "dmv": ["dmv", "dmvpay", "statemv", "ticketpay", "dmvservices"],
    "irs_refund": ["irs", "irsrefund", "taxrefund", "refundclaim", "irspay"],
    "amazon": ["amazon", "amazonrefund", "orderrefund", "amazonhelp"],
    "task_job": ["taskpay", "worktasks", "jobstart", "taskcenter"],
    "jury_duty": ["court", "uscourt", "jurypay", "warrantpay", "courtclerk"],
    "ftc_agent": ["ftc", "federaltrade", "ftchelp", "ftcrecovery"],
    "job_check": ["hronboard", "companyhr", "equipmentpay", "newhire"],
    "voter": ["voter", "votereg", "registrationcheck", "voterstatus"],
    "whatsapp_code": ["whatsapp", "verifycode", "accountverify"],
    "generic": ["secureverify", "accountverify", "payonline", "refundcenter",
                "alertcenter", "verifyaccount", "securelogin", "trackdelivery",
                "bankalert", "claimcenter"],
}
URL_SUFFIXES = ["pay", "online", "center", "help", "secure", "verify", "us",
                "now", "services", "portal"]
URL_PATHS = ["pay", "settle", "verify", "claim", "login", "resolve", "update",
             "confirm", ""]

# ---------------- slot fillers ----------------

def _money(rng, lo, hi, cents=True):
    v = rng.uniform(lo, hi)
    if cents:
        return f"{v:,.2f}"
    return f"{int(round(v)):,}"

def make_phone(rng):
    area = rng.choice(AREACODES)
    tail = rng.randint(0, 99)
    fmt = rng.randint(0, 3)
    if fmt == 0:
        return f"+1 ({area}) 555-01{tail:02d}"
    if fmt == 1:
        return f"1-{area}-555-01{tail:02d}"
    if fmt == 2:
        return f"({area}) 555-01{tail:02d}"
    return f"{area}55501{tail:02d}"

def make_date(rng):
    style = rng.randint(0, 2)
    if style == 0:
        return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}"
    if style == 1:
        return f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/2026"
    return rng.choice(WEEKDAYS)

def make_url(rng, family=None):
    cores = URL_CORES.get(family) or URL_CORES["generic"]
    if rng.random() < 0.06:
        tail = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                       for _ in range(rng.randint(5, 8)))
        return f"{rng.choice(['bit.ly', 'tinyurl.com', 't.co'])}/{tail}"
    core = rng.choice(cores)
    style = rng.randint(0, 4)
    if style == 0:
        host = f"{core}.com"
    elif style == 1:
        host = f"{core}-{rng.choice(URL_SUFFIXES)}.com"
    elif style == 2:
        host = f"www.{core}{rng.choice(URL_SUFFIXES)}.com"
    elif style == 3:
        host = f"{core}-{rng.choice(URL_SUFFIXES)}.com"
    else:
        host = f"my{core}.com"
    path = rng.choice(URL_PATHS)
    scheme = rng.choice(["https://", "https://", ""])
    return f"{scheme}{host}" + (f"/{path}" if path else "")

def _person(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

def fill_slots(text, rng, family=None):
    def amt(m):
        lo, hi = rng.choice([(3, 30), (30, 120), (80, 400), (150, 900),
                             (200, 2400)])
        return _money(rng, lo, hi)
    text = re.sub(r"\{AMT\}", amt, text)
    text = text.replace("{FEE}", _money(rng, 15, 75))
    text = text.replace("{DEP}", _money(rng, 50, 500, cents=False))
    text = text.replace("{RATE}", str(rng.choice(
        [200, 250, 300, 320, 350, 400, 450, 480, 500])))
    text = text.replace("{AGENCY}", rng.choice(TOLL_AGENCIES))
    text = text.replace("{AGENCY_TAX}",
                        rng.choice(["IRS", "Internal Revenue Service"]))
    text = text.replace("{STATE}", rng.choice(STATES))
    text = text.replace("{COUNTY}", rng.choice(COUNTIES))
    text = text.replace("{NAME}", rng.choice(FIRST_NAMES))
    text = text.replace("{DAY}", rng.choice(WEEKDAYS))
    text = text.replace("{YEAR}", rng.choice(["2025", "2026"]))
    text = text.replace("{DATE}", make_date(rng))
    text = text.replace("{PHONE}", make_phone(rng))
    text = text.replace("{BADGE}", str(rng.randint(1000, 999999)))
    text = text.replace("{ORDER}", f"{rng.randint(100, 999)}-"
                        f"{rng.randint(1000000, 9999999)}-"
                        f"{rng.randint(1000000, 9999999)}")
    text = re.sub(r"\{URL\}", lambda m: make_url(rng, family), text)
    return text

_PREFIXES = ["", "", "", "", "FINAL NOTICE: ", "URGENT: ", "Alert: ",
             "Reminder: "]

def surface_transform(text, rng):
    """Mechanical variety only: prefix label, trailing punctuation."""
    p = rng.choice(_PREFIXES)
    if p and not text.startswith(("FINAL", "URGENT", "COURT", "Alert")):
        text = p + text
    r = rng.random()
    if r < 0.12 and not text.endswith("!"):
        text += "!"
    elif r < 0.2 and text.endswith("."):
        text = text[:-1] + "!"
    return text

def augment_template(template_text, family, rng):
    return surface_transform(fill_slots(template_text, rng, family), rng)

# ---------------- IMC25 de-masking ----------------

def _digits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))

def _alnum(rng, n):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                   for _ in range(n))

def _demask_one(tag, rng):
    if tag == "<URL>":
        return make_url(rng, "generic")
    if tag == "<PHONE_NUMBER>":
        return make_phone(rng)
    if tag == "<DATE_TIME>":
        style = rng.randint(0, 2)
        if style == 0:
            return (f"{rng.choice(MONTHS)} {rng.randint(1, 28)} at "
                    f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} "
                    f"{rng.choice(['AM', 'PM'])}")
        if style == 1:
            return f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/2025"
        return rng.choice(WEEKDAYS)
    if tag == "<NAMED_ENTITY>":
        return rng.choice(ORGS + FIRST_NAMES)
    if tag == "<LOCATION>":
        return rng.choice(CITIES)
    if tag == "<EMAIL_ADDRESS>":
        return (f"{_alnum(rng, rng.randint(5, 9))}@{rng.choice(URL_CORES['generic'])}"
                f".com")
    if tag == "<US_SSN>":
        return f"9{rng.randint(0, 99):02d}-{rng.randint(10, 99)}-{_digits(rng, 4)}"
    if tag == "<US_BANK_NUMBER>":
        return _digits(rng, rng.randint(8, 12))
    if tag == "<US_DRIVER_LICENSE>":
        return f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{_digits(rng, 8)}"
    if tag == "<US_PASSPORT>":
        return f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{_digits(rng, 8)}"
    if tag == "<CREDIT_CARD>":
        return f"**** **** **** {_digits(rng, 4)}"
    if tag == "<IBAN_CODE>":
        return f"US{_digits(rng, 18)}"
    if tag == "<IP_ADDRESS>":
        return ".".join(str(rng.randint(11, 250)) for _ in range(4))
    if tag == "<MEDICAL_LICENSE>":
        return f"MD{_digits(rng, 7)}"
    if tag == "<CRYPTO>":
        return f"bc1{_alnum(rng, 20)}"
    if tag == "<NUMERO_COLIS>":
        return _alnum(rng, 12).upper()
    # <NRP>, <UK_NHS>, <ADV>, <UNBW> and any other masked identifier
    return _digits(rng, rng.randint(6, 10))

_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+>")

def demask(text, rng):
    """Replace every IMC25 researcher placeholder with a code-generated
    raw form. Deterministic for a given seeded rng."""
    return _PLACEHOLDER_RE.sub(lambda m: _demask_one(m.group(0), rng), text)
