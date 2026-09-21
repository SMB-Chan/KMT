"""Generate the KMT operator-cockpit application icon.

The icon is drawn programmatically so that it is reproducible bit for
bit on any machine that has Pillow installed::

    python3 make_app_icon.py                 # -> operator_training/assets/
    python3 make_app_icon.py --size 512 --out /tmp/icon.png

Design (top-down cockpit view, no text so no font is required):

* rounded-square sea gradient background with faint wave streaks,
* a HUD reticle (circle, degree ticks, centre crosshair),
* a flying-boat silhouette seen from above with two propeller discs.

Sizes are all fractions of the canvas, so any ``--size`` renders the
same artwork.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

SUPERSAMPLE = 4

# (stop_fraction, rgb) pairs for the vertical sea gradient.
GRADIENT = (
    (0.00, (6, 26, 44)),
    (0.45, (12, 62, 88)),
    (0.78, (16, 96, 116)),
    (1.00, (10, 72, 92)),
)

HULL = (236, 246, 252)
HULL_SHADE = (168, 198, 214)
ACCENT = (255, 176, 71)
RETICLE = (190, 232, 248)
WAVE = (146, 214, 236)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _gradient_color(y_frac: float) -> tuple:
    """Interpolate the background gradient at a normalised height."""
    stops = GRADIENT
    if y_frac <= stops[0][0]:
        return stops[0][1]
    for (y0, c0), (y1, c1) in zip(stops, stops[1:]):
        if y_frac <= y1:
            t = (y_frac - y0) / (y1 - y0)
            return tuple(int(round(_lerp(c0[i], c1[i], t))) for i in range(3))
    return stops[-1][1]


def _bbox(x0: float, y0: float, x1: float, y1: float) -> tuple:
    """Order a bounding box; mirrored artwork supplies y1 < y0."""
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _paint_background(size: int) -> Image.Image:
    """Sea gradient clipped to a rounded square."""
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(bg)
    for y in range(size):
        draw.line([(0, y), (size, y)], fill=_gradient_color(y / (size - 1)))
    mask = _rounded_mask(size, int(size * 0.21))
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)
    return out


def _paint_waves(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Faint horizontal wave streaks (top-down sea texture)."""
    for i, y_frac in enumerate((0.62, 0.72, 0.82, 0.91)):
        y0 = y_frac * size
        amp = size * (0.012 + 0.004 * i)
        phase = i * 1.7
        pts = []
        x = 0.0
        while x <= size:
            pts.append((x, y0 + amp * math.sin(2 * math.pi * (x / (size * 0.42))
                                                 + phase)))
            x += size / 240.0
        draw.line(pts, fill=WAVE + (46 + 8 * i,), width=max(1, size // 180))


def _paint_reticle(draw: ImageDraw.ImageDraw, size: int,
                   cx: float, cy: float, r: float) -> None:
    """HUD ring with degree ticks and a centre crosshair."""
    width = max(1, size // 110)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                 outline=RETICLE + (150,), width=width)
    draw.ellipse((cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62),
                 outline=RETICLE + (70,), width=width)
    for deg in range(0, 360, 15):
        long_tick = deg % 90 == 0
        inner = r * (0.86 if long_tick else 0.93)
        a = math.radians(deg)
        draw.line(
            [(cx + inner * math.cos(a), cy + inner * math.sin(a)),
             (cx + r * math.cos(a), cy + r * math.sin(a))],
            fill=RETICLE + (200 if long_tick else 110,),
            width=max(1, size // 128) if long_tick else max(1, size // 200),
        )
    gap = size * 0.022
    arm = size * 0.055
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        draw.line([(cx + dx * gap, cy + dy * gap),
                   (cx + dx * (gap + arm), cy + dy * (gap + arm))],
                  fill=ACCENT + (230,), width=max(1, size // 128))


def _paint_aircraft(draw: ImageDraw.ImageDraw, size: int,
                    cx: float, cy: float) -> None:
    """Flying boat seen from above: hull, swept wing, tail, props."""
    unit = size * 0.01

    # Hull: nose to the right, planing step and tail to the left.
    hull = [
        (cx + 20 * unit, cy),
        (cx + 15 * unit, cy - 2.4 * unit),
        (cx + 2 * unit, cy - 3.0 * unit),
        (cx - 16 * unit, cy - 2.0 * unit),
        (cx - 20 * unit, cy),
        (cx - 16 * unit, cy + 2.0 * unit),
        (cx + 2 * unit, cy + 3.0 * unit),
        (cx + 15 * unit, cy + 2.4 * unit),
    ]
    draw.polygon(hull, fill=HULL + (255,))
    draw.line([(cx - 14 * unit, cy), (cx + 16 * unit, cy)],
              fill=HULL_SHADE + (255,), width=max(1, int(0.9 * unit)))

    # Main wing: swept, tapered, spanning vertically in this view.
    for sign in (-1, 1):
        wing = [
            (cx + 1 * unit, cy + sign * 2.0 * unit),
            (cx - 3 * unit, cy + sign * 26 * unit),
            (cx - 6.5 * unit, cy + sign * 26 * unit),
            (cx - 6 * unit, cy + sign * 2.0 * unit),
        ]
        draw.polygon(wing, fill=HULL + (255,))
        # Outrigger sponson near the wing tip.
        draw.rounded_rectangle(
            _bbox(cx - 8 * unit, cy + sign * 20 * unit,
                  cx - 3 * unit, cy + sign * 23 * unit),
            radius=int(unit), fill=HULL_SHADE + (255,))

    # Tail plane.
    for sign in (-1, 1):
        draw.polygon([
            (cx - 15 * unit, cy + sign * 1.4 * unit),
            (cx - 17 * unit, cy + sign * 9 * unit),
            (cx - 19.5 * unit, cy + sign * 9 * unit),
            (cx - 18 * unit, cy + sign * 1.4 * unit),
        ], fill=HULL + (255,))
    # Vertical fin seen edge-on.
    draw.polygon([
        (cx - 15 * unit, cy - 0.8 * unit),
        (cx - 18 * unit, cy - 0.8 * unit),
        (cx - 20 * unit, cy - 6 * unit),
        (cx - 17.5 * unit, cy - 6 * unit),
    ], fill=ACCENT + (255,))

    # Propeller discs on the wing, with a spinner hub.
    pr = 4.6 * unit
    for sign in (-1, 1):
        px, py = cx - 2 * unit, cy + sign * 15 * unit
        draw.ellipse((px - pr, py - pr, px + pr, py + pr),
                     outline=ACCENT + (235,), width=max(1, int(0.7 * unit)))
        blade = pr * 0.92
        for a in (0.0, math.pi / 2.0):
            draw.line([(px - blade * math.cos(a), py - blade * math.sin(a)),
                       (px + blade * math.cos(a), py + blade * math.sin(a))],
                      fill=RETICLE + (170,), width=max(1, int(0.6 * unit)))
        hub = 1.3 * unit
        draw.ellipse((px - hub, py - hub, px + hub, py + hub),
                     fill=HULL_SHADE + (255,))


def render_icon(size: int = 256) -> Image.Image:
    """Return the icon as an RGBA ``Image`` of ``size`` x ``size`` pixels."""
    if size < 16:
        raise ValueError("icon size must be >= 16 px")
    s = size * SUPERSAMPLE
    canvas = _paint_background(s)
    draw = ImageDraw.Draw(canvas, "RGBA")
    _paint_waves(draw, s)
    cx, cy = s * 0.5, s * 0.47
    _paint_reticle(draw, s, cx, cy, s * 0.335)
    _paint_aircraft(draw, s, cx, cy)
    return canvas.resize((size, size), Image.LANCZOS)


def write_icon(path: Path, size: int = 256) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    render_icon(size).save(path, format="PNG", optimize=True)
    return path


ICON_SIZES = (256, 128, 64, 48)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="write a single icon to this path")
    parser.add_argument("--size", type=int, default=256,
                        help="icon edge length in px (default 256)")
    parser.add_argument("--all-sizes", action="store_true",
                        help="write every size in ICON_SIZES next to --out")
    args = parser.parse_args(argv)

    out = args.out or (Path(__file__).resolve().parent
                       / "operator_training" / "assets"
                       / "kmt-operator-cockpit.png")
    if args.all_sizes:
        stem = out.with_suffix("")
        for size in ICON_SIZES:
            p = out if size == 256 else stem.with_name(f"{stem.name}-{size}.png")
            write_icon(p, size)
            print(f"wrote {p} ({size}x{size})")
    else:
        write_icon(out, args.size)
        print(f"wrote {out} ({args.size}x{args.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
