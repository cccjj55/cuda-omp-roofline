#!/usr/bin/env python3
"""Render docs/roofline.svg from verified CPU & GPU kernel measurements in README.md."""
from __future__ import annotations

import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def render(out_path: pathlib.Path) -> None:
    # Verified operational intensities (FLOP/Byte, compulsory DRAM traffic) and achieved GFLOP/s:
    # axpy (double): 2 FLOP / (3*8 B) = 0.0833 FLOP/B -> CPU 58.35 GB/s = 4.86 GFLOP/s, GPU 402.4 GB/s = 33.53 GFLOP/s
    # reduce (double): 1 FLOP / 8 B = 0.125 FLOP/B -> GPU 417.9 GB/s = 52.24 GFLOP/s
    # softmax (double, 2-pass): ~7 FLOP / (4*8 B) = 0.219 FLOP/B -> CPU 1.20 Gelem/s (~8.4 GFLOP/s), GPU 8.65 Gelem/s (~60.6 GFLOP/s)
    # matmul tiled 64 (double, CPU): ~2*512^3 / (DRAM bytes; L1/L2 reuse ~ 2*64/24 = 5.33) -> 70.62 GFLOP/s
    # matmul tiled 32 (double, GPU, smem reuse 32): 2*32^3 / (2*32*32*8) = 4.0 FLOP/B -> 444.42 GFLOP/s
    # matmul tiled 32 (float, GPU, smem reuse 32): 2*32^3 / (2*32*32*4) = 8.0 FLOP/B -> 2285.91 GFLOP/s
    points = [
        ("axpy CPU (12t)", 0.0833, 4.86, "#58a6ff"),
        ("axpy GPU", 0.0833, 33.53, "#56d364"),
        ("reduce GPU (shfl)", 0.125, 52.24, "#56d364"),
        ("softmax GPU (grid)", 0.219, 60.57, "#56d364"),
        ("matmul CPU (tiled+omp)", 4.0, 70.62, "#58a6ff"),
        ("matmul GPU (fp64)", 4.0, 444.42, "#f0883e"),
        ("matmul GPU (fp32)", 8.0, 2285.91, "#a371f7"),
    ]

    w, h = 920, 390
    gx, gy, gw, gh = 86, 76, 780, 260
    min_x, max_x = math.log10(0.05), math.log10(16.0)
    min_y, max_y = math.log10(2.0), math.log10(5000.0)

    def sx(ai: float) -> float:
        return gx + (math.log10(ai) - min_x) / (max_x - min_x) * gw

    def sy(gf: float) -> float:
        return gy + gh - (math.log10(gf) - min_y) / (max_y - min_y) * gh

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" rx="12" fill="#0d1117"/>',
        '<text x="32" y="34" fill="#f0f6fc" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700">'
        'Measured Arithmetic Intensity vs Performance (i7-14650HX 12t vs RTX 5070 Ti sm_120)</text>',
        '<text x="32" y="54" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="12">'
        'Log-log view: axpy/reduce/softmax sit on the DRAM slope (58.4 GB/s CPU vs 402 GB/s GPU); FP64 matmul hits cut FP64 pipe (5.1x below FP32)</text>',
        f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="6" fill="#161b22" stroke="#30363d"/>',
    ]

    for ytick in (5, 10, 50, 100, 500, 1000, 3000):
        ty = sy(float(ytick))
        lines.append(f'<line x1="{gx}" y1="{ty:.1f}" x2="{gx + gw}" y2="{ty:.1f}" stroke="#21262d"/>')
        lines.append(
            f'<text x="{gx - 8}" y="{ty + 4:.1f}" text-anchor="end" fill="#8b949e" '
            f'font-family="Consolas, monospace" font-size="10">{ytick}</text>'
        )
    for xtick in (0.08, 0.25, 1.0, 4.0, 8.0):
        tx = sx(xtick)
        lines.append(f'<line x1="{tx:.1f}" y1="{gy}" x2="{tx:.1f}" y2="{gy + gh}" stroke="#21262d"/>')
        lines.append(
            f'<text x="{tx:.1f}" y="{gy + gh + 18}" text-anchor="middle" fill="#8b949e" '
            f'font-family="Consolas, monospace" font-size="10">{xtick:g} F/B</text>'
        )

    # Bandwidth slopes: CPU 58.35 GB/s, GPU 402.4 GB/s
    for bw, col, lbl, lx in [
        (58.35, "#58a6ff", "CPU DRAM wall (58.4 GB/s)", 1.5),
        (402.4, "#56d364", "GPU DRAM wall (402.4 GB/s)", 1.2),
    ]:
        x1, x2 = 0.05, 8.0
        y1, y2 = max(2.0, bw * x1), min(4800.0, bw * x2)
        if bw * x2 > 4800.0:
            x2 = 4800.0 / bw
            y2 = 4800.0
        lines.append(
            f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" '
            f'stroke="{col}" stroke-dasharray="5,5" stroke-opacity="0.65" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{sx(lx):.1f}" y="{sy(bw * lx) - 8:.1f}" fill="{col}" '
            f'font-family="Consolas, monospace" font-size="10">{lbl}</text>'
        )

    # Compute ceilings from measured tiled matmul
    for gf, col, lbl in [(2285.91, "#a371f7", "FP32 tiled peak (2,286 GFLOP/s)"), (444.42, "#f0883e", "FP64 tiled peak (444 GFLOP/s)")]:
        ty = sy(gf)
        lines.append(
            f'<line x1="{sx(1.0):.1f}" y1="{ty:.1f}" x2="{gx + gw}" y2="{ty:.1f}" '
            f'stroke="{col}" stroke-dasharray="3,4" stroke-opacity="0.6" stroke-width="1.2"/>'
        )
        lines.append(
            f'<text x="{gx + gw - 8}" y="{ty - 6:.1f}" text-anchor="end" fill="{col}" '
            f'font-family="Consolas, monospace" font-size="10">{lbl}</text>'
        )

    # Points with explicit label offsets to prevent overlap
    offsets = {
        "axpy CPU (12t)": (8, 12, "start"),
        "axpy GPU": (8, 16, "start"),
        "reduce GPU (shfl)": (-10, -14, "middle"),
        "softmax GPU (grid)": (8, 16, "start"),
        "matmul CPU (tiled+omp)": (-8, 16, "end"),
        "matmul GPU (fp64)": (-8, 16, "end"),
        "matmul GPU (fp32)": (-8, 16, "end"),
    }
    for name, ai, gf, col in points:
        px, py = sx(ai), sy(gf)
        dx, dy, anchor = offsets[name]
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}" stroke="#0d1117" stroke-width="1.5"/>')
        lines.append(
            f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}" fill="#e6edf3" '
            f'font-family="Consolas, monospace" font-size="10">{name} ({gf:.1f})</text>'
        )

    lines.append(
        f'<text x="{gx + gw / 2:.1f}" y="{gy + gh + 36}" text-anchor="middle" fill="#8b949e" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="11">Arithmetic Intensity (FLOP / Byte)</text>'
    )
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render(ROOT / "docs" / "roofline.svg")
