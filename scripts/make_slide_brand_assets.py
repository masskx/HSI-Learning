"""Render the course identity graphic used on deck covers and closing slides.

The graphic is the real mean spectrum of labelled Indian Pines pixels
(video/intro/spectrum.json, written by build_video_intro.py) drawn as 200
spectral-coloured band bars with the curve on top -- the same motif the video
intro animates, so the intro's last frame and the deck cover match.

    python scripts/make_slide_brand_assets.py
"""
from pathlib import Path
import json

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'slides' / 'assets' / 'brand'
STOPS = ['472A8A', '2C5FB0', '1F8FB8', '2E9E4F', 'F2C316', 'E8752A', 'D33A2C']


def spectral(u):
    x = max(0.0, min(1.0, u)) * (len(STOPS) - 1)
    i = min(int(x), len(STOPS) - 2)
    f = x - i
    a = [int(STOPS[i][k:k + 2], 16) for k in (0, 2, 4)]
    b = [int(STOPS[i + 1][k:k + 2], 16) for k in (0, 2, 4)]
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def bars(width=2667, height=320, ss=2, bar_alpha=150, line=True):
    spec = json.loads((ROOT / 'video' / 'intro' / 'spectrum.json').read_text(encoding='utf-8'))
    w, h = width * ss, height * ss
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    n = len(spec)
    pitch = w / n
    top = lambda v: h - (0.14 + 0.82 * v) * h
    for i, v in enumerate(spec):
        x0 = i * pitch + pitch * 0.18
        draw.rectangle([x0, top(v), x0 + pitch * 0.42, h], fill=spectral(i / (n - 1)) + (bar_alpha,))
    if line:
        pts = [((i + 0.39) * pitch, top(v)) for i, v in enumerate(spec)]
        for i in range(n - 1):
            draw.line([pts[i], pts[i + 1]], fill=spectral(i / (n - 1)) + (255,), width=4 * ss)
    return img.resize((width, height), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bars().save(OUT / 'spectrum-bars.png', optimize=True)
    print('wrote', OUT / 'spectrum-bars.png')


if __name__ == '__main__':
    main()
