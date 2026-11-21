#!/usr/bin/env python3
"""Code-drawn promo card for Scam Shield, 1200x630. No AI imagery."""
from PIL import Image, ImageDraw, ImageFont
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scam-shield-promo-v4.png")
W, H = 1200, 630

NOTO = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
NOTOR = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
if not os.path.exists(NOTO):
    NOTO = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    NOTOR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def F(path, size):
    return ImageFont.truetype(path, size)

# palette
NAVY_TOP = (9, 32, 48)
NAVY_BOT = (13, 58, 74)
WHITE = (255, 255, 255)
SOFT = (203, 228, 238)
MUTED = (148, 183, 199)
AMBER = (245, 176, 65)
TEAL_CHIP = (20, 74, 96)
RED = (208, 48, 49)
GREEN = (34, 150, 84)
SCREEN = (244, 248, 249)
INK = (30, 44, 52)
GRAYTXT = (84, 102, 112)

# ---- background: vertical gradient navy -> teal
img = Image.new("RGB", (W, H))
px = img.load()
for y in range(H):
    t = y / (H - 1)
    c = tuple(int(NAVY_TOP[i] + (NAVY_BOT[i] - NAVY_TOP[i]) * t) for i in range(3))
    for x in range(W):
        px[x, y] = c
d = ImageDraw.Draw(img, "RGBA")

# soft amber glow behind shield area (heavily blurred, no hard edge)
from PIL import ImageFilter
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([-140, -180, 420, 380], fill=(245, 176, 65, 34))
glow = glow.filter(ImageFilter.GaussianBlur(90))
img = Image.alpha_composite(img.convert("RGBA"), glow)
d = ImageDraw.Draw(img, "RGBA")

def shield(cx, cy, w, h, fill, check_color, lw_ratio=0.13):
    """Draw a shield centered at cx, top at cy."""
    import math
    pts = []
    # top-left -> top-right
    pts.append((cx - w / 2, cy + h * 0.06))
    pts.append((cx + w / 2, cy + h * 0.06))
    # right side down, curve to bottom point
    for i in range(1, 13):
        t = i / 12
        x = cx + w / 2
        y = cy + h * 0.06 + (h * 0.42) * t
        pts.append((x, y))
    for i in range(1, 17):
        t = i / 16
        # quadratic bezier from (cx+w/2, cy+h*0.48) via (cx+w/2, cy+h*0.82) to (cx, cy+h)
        x0, y0 = cx + w / 2, cy + h * 0.48
        x1, y1 = cx + w / 2, cy + h * 0.84
        x2, y2 = cx, cy + h
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2
        pts.append((x, y))
    for i in range(1, 17):
        t = i / 16
        x0, y0 = cx, cy + h
        x1, y1 = cx - w / 2, cy + h * 0.84
        x2, y2 = cx - w / 2, cy + h * 0.48
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2
        pts.append((x, y))
    for i in range(1, 13):
        t = i / 12
        pts.append((cx - w / 2, cy + h * 0.48 - (h * 0.42) * t))
    d.polygon(pts, fill=fill)
    # check mark
    lw = max(6, int(w * lw_ratio))
    p1 = (cx - w * 0.26, cy + h * 0.46)
    p2 = (cx - w * 0.07, cy + h * 0.65)
    p3 = (cx + w * 0.30, cy + h * 0.28)
    d.line([p1, p2], fill=check_color, width=lw, joint="curve")
    d.line([p2, p3], fill=check_color, width=lw, joint="curve")
    for p in (p1, p3):
        r = lw / 2
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=check_color)

# ---- left column
LX = 64
# shield badge: amber shield w/ navy check, left of headline
shield(112, 58, 92, 108, AMBER, NAVY_TOP)
d.text((178, 74), "Scam Shield", font=F(NOTO, 84), fill=WHITE)

d.text((LX, 196), "Paste a suspicious text.", font=F(NOTOR, 37), fill=SOFT)
d.text((LX, 242), "Get a plain answer.", font=F(NOTO, 37), fill=WHITE)

# chips
chips = [
    ("97.8%", " of modern scam texts caught"),
    ("0.44%", " false alarms on legit texts"),
    ("0.31 MB, works offline", " \u2014 nothing leaves the phone"),
]
cy = 330
f_b = F(NOTO, 27)
f_r = F(NOTOR, 27)
for bold, rest in chips:
    wb = d.textlength(bold, font=f_b)
    wr = d.textlength(rest, font=f_r)
    cw = int(wb + wr) + 68
    d.rounded_rectangle([LX, cy, LX + cw, cy + 58], radius=29, fill=TEAL_CHIP)
    d.ellipse([LX + 18, cy + 24, LX + 28, cy + 34], fill=AMBER)
    d.text((LX + 40, cy + 13), bold, font=f_b, fill=AMBER)
    d.text((LX + 40 + wb, cy + 13), rest, font=f_r, fill=WHITE)
    cy += 74

d.text((LX, 566), "Free \u00b7 Open source \u00b7 github.com/RustyDrops/Scam_Shield_Tiny_ML",
       font=F(NOTOR, 20), fill=MUTED)

# ---- phone mockup (right)
PX0, PY0, PX1, PY1 = 806, 42, 1146, 588
d.rounded_rectangle([PX0 - 8, PY0 - 8, PX1 + 8, PY1 + 8], radius=46, fill=(5, 18, 26))
d.rounded_rectangle([PX0, PY0, PX1, PY1], radius=38, fill=SCREEN)
# notch
d.rounded_rectangle([PX0 + 118, PY0 + 12, PX1 - 118, PY0 + 30], radius=9, fill=(5, 18, 26))
# header
d.text((PX0 + 26, PY0 + 46), "Unknown sender", font=F(NOTO, 21), fill=INK)
d.text((PX0 + 26, PY0 + 74), "checked by Scam Shield", font=F(NOTOR, 16), fill=GRAYTXT)
d.line([PX0 + 26, PY0 + 104, PX1 - 26, PY0 + 104], fill=(214, 224, 228), width=2)

def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=f_r2) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines

f_r2 = F(NOTOR, 19)
def bubble(x, y, w, text, badge_text, badge_color, is_scam):
    lines = wrap(text, f_r2, w - 32)
    bh = 16 + len(lines) * 26 + 12
    d.rounded_rectangle([x, y, x + w, y + bh], radius=16, fill=WHITE,
                        outline=(205, 216, 221), width=2)
    ty = y + 13
    for ln in lines:
        d.text((x + 16, ty), ln, font=f_r2, fill=INK)
        ty += 26
    # badge overlapping bottom-right of bubble
    f_badge = F(NOTO, 20)
    bw = int(d.textlength(badge_text, font=f_badge)) + 52
    bx1 = x + w - 6
    bx0 = bx1 - bw
    by0 = y + bh - 6
    by1 = by0 + 38
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=19, fill=badge_color)
    # icon: check or bang
    ix = bx0 + 24
    icy = (by0 + by1) / 2
    if is_scam:
        d.text((ix - 4, by0 + 6), "!", font=F(NOTO, 22), fill=WHITE)
    else:
        d.line([(ix - 9, icy), (ix - 2, icy + 7)], fill=WHITE, width=4)
        d.line([(ix - 2, icy + 7), (ix + 11, icy - 8)], fill=WHITE, width=4)
    d.text((bx0 + 40, by0 + 8), badge_text, font=f_badge, fill=WHITE)
    return by1

y_end = bubble(PX0 + 24, PY0 + 122, PX1 - PX0 - 48,
               "E-ZPass: You have an unpaid toll of $12.51. Pay now to avoid a $50 late fee: pay-ezpass-now.com",
               "LIKELY SCAM", RED, True)
y_end2 = bubble(PX0 + 24, y_end + 26, PX1 - PX0 - 48,
                "Your verification code is 482916. Don't share it with anyone.",
                "SAFE-LOOKING", GREEN, False)
cap = "Checked on this phone \u2014 nothing uploaded"
f_cap = F(NOTOR, 13)
d.text((PX0 + (PX1 - PX0 - d.textlength(cap, font=f_cap)) / 2, PY1 - 44),
       cap, font=f_cap, fill=GRAYTXT)

img.convert("RGB").save(OUT, "PNG")
print("saved", OUT, os.path.getsize(OUT), "second bubble ends at y =", y_end2, "phone bottom =", PY1)
