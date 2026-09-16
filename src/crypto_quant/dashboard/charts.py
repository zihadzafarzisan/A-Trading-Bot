"""Inline SVG chart generators.

Produces self-contained SVG markup (no JavaScript, no CDN, no network) so the
generated dashboard is fully local per the specification. Charts are crisp,
theme-aware, and render natively in any browser.

Charts: line, area (filled), bar, histogram, donut, sparkline.
"""

import html
import math
from typing import Dict, List, Optional, Sequence, Tuple


def _esc(v) -> str:
    """Escape a value for safe HTML embedding."""
    if v is None:
        return ""
    return html.escape(str(v))


def _rgb_alpha(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb + alpha to rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Shared scaling helpers
# ---------------------------------------------------------------------------
def _nice_scale(data: Sequence[float], pad: float = 0.05):
    lo, hi = min(data), max(data)
    span = hi - lo or 1.0
    return lo - span * pad, hi + span * pad


# ---------------------------------------------------------------------------
# Line / area chart
# ---------------------------------------------------------------------------
def line_chart(
    x: Sequence,
    y: Sequence[float],
    width: int = 760,
    height: int = 260,
    stroke: str = "#4f9cf9",
    fill: Optional[str] = "#4f9cf9",
    x_labels: Optional[List[str]] = None,
    y_decimals: int = 1,
    title: str = "",
) -> str:
    """A line chart (optionally area-filled) rendered as inline SVG."""
    if len(x) == 0:
        return _empty_chart(width, height, title)
    pad_l, pad_r, pad_t, pad_b = 54, 16, 30, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    lo, hi = _nice_scale([float(v) for v in y])
    def X(i): return pad_l + (i / max(1, len(x) - 1)) * plot_w
    def Y(v): return pad_t + (1 - (v - lo) / (hi - lo)) * plot_h

    pts = " ".join(f"{X(i):.1f},{Y(float(v)):.1f}" for i, v in enumerate(y))
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']

    # grid + y labels
    for k in range(5):
        gy = pad_t + k * plot_h / 4
        val = hi - k * (hi - lo) / 4
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
                     f'stroke="#2a3350" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" text-anchor="end" '
                     f'fill="#8a94b8" font-size="10">{val:.{y_decimals}f}</text>')
    # x labels
    if x_labels:
        step = max(1, len(x_labels) // 8)
        for i in range(0, len(x_labels), step):
            parts.append(f'<text x="{X(i):.1f}" y="{height - 12}" text-anchor="middle" '
                         f'fill="#8a94b8" font-size="10">{_esc(x_labels[i])}</text>')
    # area
    if fill:
        area = f"{pts} L{X(len(y) - 1):.1f},{pad_t + plot_h:.1f} L{X(0):.1f},{pad_t + plot_h:.1f} Z"
        parts.append(f'<path d="{area}" fill="{_rgb_alpha(fill, 0.18)}" stroke="none"/>')
    # line
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    # hover targets (invisible tall points with <title>)
    for i, v in enumerate(y):
        parts.append(f'<circle cx="{X(i):.1f}" cy="{Y(float(v)):.1f}" r="3" fill="transparent">'
                     f'<title>{(x_labels[i] if x_labels else str(x[i]))}: {v:.{y_decimals}f}</title></circle>')
    if title:
        parts.append(f'<text x="{pad_l}" y="16" fill="#c8d2f0" font-size="13" font-weight="600">{_esc(title)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    width: int = 760,
    height: int = 240,
    color: str = "#4f9cf9",
    value_format: str = "{:.2f}",
    title: str = "",
) -> str:
    """Vertical bar chart with labels and value tooltips."""
    if not labels:
        return _empty_chart(width, height, title)
    pad_l, pad_r, pad_t, pad_b = 44, 10, 26, 42
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = 0.0, max([float(v) for v in values] + [0.0]) * 1.05

    def Y(v): return pad_t + (1 - v / hi) * plot_h if hi else pad_t + plot_h

    n = len(labels)
    slot = plot_w / n
    bw = max(6.0, slot * 0.62)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']
    for k in range(4):
        gy = pad_t + k * plot_h / 3
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
                     f'stroke="#2a3350" stroke-width="1"/>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        x0 = pad_l + i * slot + (slot - bw) / 2
        y0 = Y(float(v))
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{pad_t + plot_h - y0:.1f}" '
                     f'rx="2" fill="{color}"><title>{_esc(lab)}: {value_format.format(v)}</title></rect>')
        # label (rotated if long)
        lab_text = _esc(str(lab))
        if len(lab_text) > 10:
            parts.append(f'<text x="{x0 + bw / 2:.1f}" y="{height - 8}" text-anchor="end" '
                         f'fill="#8a94b8" font-size="9" transform="rotate(-35 {x0 + bw / 2:.1f} {height - 8})">{lab_text}</text>')
        else:
            parts.append(f'<text x="{x0 + bw / 2:.1f}" y="{height - 14}" text-anchor="middle" '
                         f'fill="#8a94b8" font-size="10">{lab_text}</text>')
    if title:
        parts.append(f'<text x="{pad_l}" y="16" fill="#c8d2f0" font-size="13" font-weight="600">{_esc(title)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def histogram(
    values: Sequence[float],
    bins: int = 20,
    width: int = 760,
    height: int = 240,
    color: str = "#7fd08a",
    title: str = "",
) -> str:
    """Histogram of a numeric distribution."""
    vals = [float(v) for v in values]
    if not vals:
        return _empty_chart(width, height, title)
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1.0
    hist = [0] * bins
    for v in vals:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        hist[idx] += 1
    labels = [f"{lo + (i + 0.5) * (hi - lo) / bins:.2f}" for i in range(bins)]
    return bar_chart(labels, hist, width=width, height=height, color=color,
                     value_format="{:.0f}", title=title)


# ---------------------------------------------------------------------------
# Donut chart
# ---------------------------------------------------------------------------
def donut_chart(
    labels: Sequence[str],
    values: Sequence[float],
    width: int = 320,
    height: int = 260,
    colors: Optional[List[str]] = None,
    title: str = "",
) -> str:
    """Donut chart for categorical shares."""
    vals = [float(v) for v in values]
    total = sum(vals)
    if total <= 0:
        return _empty_chart(width, height, title)
    palette = colors or ["#4f9cf9", "#7fd08a", "#f9c74f", "#f4845f",
                         "#9b8afb", "#f2778a", "#5fb7b7", "#d0a0f0"]
    cx, cy, r, ir = width / 2, height / 2, min(width, height) / 2 - 30, 34
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']
    start = -90.0
    for i, (lab, v) in enumerate(zip(labels, vals)):
        frac = v / total
        sweep = frac * 360
        end = start + sweep
        large = 1 if sweep > 180 else 0
        x1 = cx + r * math.cos(math.radians(start))
        y1 = cy + r * math.sin(math.radians(start))
        x2 = cx + r * math.cos(math.radians(end))
        y2 = cy + r * math.sin(math.radians(end))
        color = palette[i % len(palette)]
        d = (f"M {cx:.1f} {cy:.1f} L {x1:.1f} {y1:.1f} A {r:.1f} {r:.1f} 0 {large} 1 "
             f"{x2:.1f} {y2:.1f} Z")
        parts.append(f'<path d="{d}" fill="{color}" stroke="#0d1326" stroke-width="1">'
                     f'<title>{_esc(lab)}: {frac:.1%}</title></path>')
        start = end
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{ir}" fill="#0d1326"/>')
    parts.append(f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" fill="#c8d2f0" '
                 f'font-size="16" font-weight="700">{total:.0f}</text>')
    parts.append(f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" fill="#8a94b8" '
                 f'font-size="10">total</text>')
    if title:
        parts.append(f'<text x="{cx}" y="18" text-anchor="middle" fill="#c8d2f0" '
                     f'font-size="13" font-weight="600">{_esc(title)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sparkline(y: Sequence[float], width: int = 180, height: int = 40, color: str = "#4f9cf9") -> str:
    """Tiny sparkline for summary cards."""
    if not y:
        return ""
    lo, hi = _nice_scale([float(v) for v in y], pad=0.1)
    def X(i): return (i / max(1, len(y) - 1)) * width
    def Y(v): return height - (v - lo) / (hi - lo) * height
    pts = " ".join(f"{X(i):.1f},{Y(float(v)):.1f}" for i, v in enumerate(y))
    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">' \
           f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'


def _empty_chart(width: int, height: int, title: str) -> str:
    """Placeholder when there is no data to chart."""
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" fill="#8a94b8" '
            f'font-size="12">{_esc(title or "No data")}</text></svg>')