"""Teaser: Fast & Furious review, three documents, last call + short answer."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

W = 1800
BG = (255, 255, 255)
INK = (22, 22, 22)
SLATE = (92, 96, 104)
RULE = (210, 212, 216)
PASS = (110, 158, 118)
PASS_INK = (78, 122, 88)
PASS_FILL = (236, 245, 237)
FAIL = (196, 118, 118)
FAIL_INK = (168, 90, 90)
FAIL_FILL = (252, 240, 240)
CODE = (32, 34, 38)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "teaser.png"
WIN = Path(r"C:\Windows\Fonts")

MARGIN = 56
GAP = 40
# DeepSeek V4 Flash, TMDB seed 9, query 38. Locked traces.
QUERY = (
    "Give me a review of a movie",
    "from the collection The Fast and the Furious.",
)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = WIN / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(str(WIN / "arial.ttf"), size)


F_QUERY = font("georgia.ttf", 42)
F_HEAD = font("segoeuib.ttf", 34)
F_CODE = font("consola.ttf", 26)
F_SLOT = font("consolab.ttf", 24)
F_META = font("segoeui.ttf", 24)
F_ANSWER = font("segoeuib.ttf", 26)

Part = str | tuple[str, str, bool]


def slot(draw: ImageDraw.ImageDraw, x: int, mid_y: int, text: str, filled: bool) -> int:
    pad_x = 9
    tw = int(draw.textlength(text, font=F_SLOT))
    box_w = tw + 2 * pad_x
    box_h = 40
    box = (x, mid_y - box_h // 2, x + box_w, mid_y + box_h // 2)
    if filled:
        draw.rounded_rectangle(box, radius=6, fill=PASS_FILL, outline=PASS, width=3)
        draw.text((x + pad_x, mid_y), text, font=F_SLOT, fill=PASS_INK, anchor="lm")
    else:
        draw.rounded_rectangle(box, radius=6, fill=FAIL_FILL, outline=FAIL, width=3)
        draw.text((x + pad_x, mid_y), text, font=F_SLOT, fill=FAIL_INK, anchor="lm")
    return x + box_w


ICON = 28


def text_mid_y(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fnt) -> float:
    box = draw.textbbox((x, y), text, font=fnt, anchor="la")
    return (box[1] + box[3]) / 2


def mark_fail(draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    s = 6.5
    draw.line((cx - s, cy - s, cx + s, cy + s), fill=FAIL, width=3)
    draw.line((cx - s, cy + s, cx + s, cy - s), fill=FAIL, width=3)


def mark_pass(draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    draw.line((cx - 6, cy + 1, cx - 1, cy + 6), fill=PASS, width=3)
    draw.line((cx - 1, cy + 6, cx + 7, cy - 6), fill=PASS, width=3)


def status_row(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fnt,
    ok: bool,
    col_w: int,
) -> int:
    color = PASS_INK if ok else FAIL_INK
    chunks = wrap_text(draw, text, col_w - ICON - 8, fnt)
    mid = text_mid_y(draw, x + ICON, y, chunks[0], fnt)
    cx = x + ICON / 2
    if ok:
        mark_pass(draw, cx, mid)
    else:
        mark_fail(draw, cx, mid)
    draw.text((x + ICON, y), chunks[0], font=fnt, fill=color, anchor="la")
    y += 32
    for chunk in chunks[1:]:
        draw.text((x + ICON, y), chunk, font=fnt, fill=color, anchor="la")
        y += 28
    return y


def parts_line(draw: ImageDraw.ImageDraw, x: int, mid_y: int, parts: Sequence[Part]) -> None:
    cx = x
    for part in parts:
        if isinstance(part, str):
            draw.text((cx, mid_y), part, font=F_CODE, fill=CODE, anchor="lm")
            cx += int(draw.textlength(part, font=F_CODE))
        else:
            _, text, filled = part
            cx = slot(draw, cx + 3, mid_y, text, filled) + 3


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_w: int, fnt) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def main() -> None:
    col_w = (W - 2 * MARGIN - 2 * GAP) // 3
    columns = [
        {
            "title": "Catalog",
            "lines": [
                ["GET /search/collection  →  9485"],
                ["GET /collection/", ("slot", "9485", True)],
                ["GET /movie/", ("slot", "movie_id", False), "/reviews"],
            ],
            "ret": "missing path parameter movie_id",
            "ans": "No review.",
            "ok": False,
        },
        {
            "title": "DRAFT",
            "lines": [
                ["GET /search/collection"],
                ["GET /collection/", ("slot", "collection_id", False)],
                ["GET /movie/", ("slot", "movie_id", False), "/reviews"],
            ],
            "ret": "missing path parameter movie_id",
            "ans": "No review.",
            "ok": False,
        },
        {
            "title": "Ours",
            "lines": [
                ["GET /search/collection  →  9485"],
                ["GET /collection/", ("slot", "9485", True)],
                ["GET /movie/", ("slot", "584", True), "/reviews"],
            ],
            "ret": "200  2 Fast 2 Furious",
            "ans": "6/10 — John Chard.",
            "ok": True,
        },
    ]

    hop_h = 54
    hops = 3
    h = 200 + 48 + hops * hop_h + 180
    img = Image.new("RGB", (W, h), BG)
    draw = ImageDraw.Draw(img)

    draw.text((W // 2, 44), QUERY[0], font=F_QUERY, fill=INK, anchor="ma")
    draw.text((W // 2, 100), QUERY[1], font=F_QUERY, fill=INK, anchor="ma")
    draw.line((MARGIN, 168, W - MARGIN, 168), fill=INK, width=2)

    top = 204
    for i, col in enumerate(columns):
        x = MARGIN + i * (col_w + GAP)
        y = top
        draw.text((x, y), col["title"], font=F_HEAD, fill=INK, anchor="la")
        y += 40
        draw.line((x, y, x + col_w, y), fill=RULE, width=2)
        y += 34
        for line in col["lines"]:
            parts_line(draw, x, y, line)
            y += hop_h
        y += 6
        draw.line((x, y, x + col_w, y), fill=RULE, width=2)
        y += 28
        y = status_row(draw, x, y, col["ret"], F_META, col["ok"], col_w)
        y = status_row(draw, x, y, col["ans"], F_ANSWER, col["ok"], col_w)

    img.save(OUT, "PNG")
    print(OUT, img.size)


if __name__ == "__main__":
    main()
