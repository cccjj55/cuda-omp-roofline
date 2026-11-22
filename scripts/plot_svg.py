#!/usr/bin/env python3
"""Render docs/kernels.svg from the measured tables in README.md."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _parse_matmul(readme: str) -> list[tuple[str, float, float, float]]:
    ci_rows: dict[str, tuple[float, float]] = {}
    laptop_rows: dict[str, float] = {}
    section = None
    for line in readme.splitlines():
        if "| kernel | variant | run A" in line:
            section = "ci"
        elif "| kernel | variant | metric | spread |" in line:
            section = "laptop"
        elif not line.startswith("|") and line.strip():
            section = None
        elif section and line.startswith("|"):
            cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
            if cells[0] != "matmul":
                continue
            variant = cells[1]
            if section == "ci":
                ci_rows[variant] = (float(cells[2].split()[0]), float(cells[3].split()[0]))
            elif section == "laptop":
                laptop_rows[variant] = float(cells[2].split()[0])

    out = []
    for v in ("naive", "tiled 64", "tiled 64 + omp"):
        a, b = ci_rows[v]
        c = laptop_rows[v]
        out.append((v, a, b, c))
    return out


def render(readme_path: pathlib.Path, out_path: pathlib.Path) -> None:
    matmul = _parse_matmul(readme_path.read_text(encoding="utf-8"))
    w, h = 920, 340
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" rx="12" fill="#0d1117"/>',
        '<text x="32" y="36" fill="#f0f6fc" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700">'
        'hpc-kernels: Cache Blocking &amp; OpenMP Scaling (matmul n=512, GFLOP/s)</text>',
        '<text x="32" y="58" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="12">'
        'GitHub CI (4 vCPU Xeon 8573C, Run A &amp; Run B) vs WSL2 Laptop (i7-14650HX, 12 threads) · best of 5</text>',
        '<rect x="32" y="82" width="856" height="226" rx="8" fill="#161b22" stroke="#30363d"/>',
    ]

    max_val = max(max(a, b, c) for _, a, b, c in matmul)
    bx0 = 180
    bw_max = 580

    # Legend
    for lx, color, lbl in [
        (48, "#58a6ff", "CI Run A (4 vCPU)"),
        (210, "#a371f7", "CI Run B (4 vCPU)"),
        (372, "#56d364", "Laptop (12 threads)"),
    ]:
        lines.append(f'<rect x="{lx}" y="98" width="12" height="12" rx="2" fill="{color}"/>')
        lines.append(
            f'<text x="{lx + 18}" y="108" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">{lbl}</text>'
        )

    for idx, (variant, a, b, c) in enumerate(matmul):
        y = 130 + idx * 56
        lines.append(
            f'<text x="{bx0 - 12}" y="{y + 26}" text-anchor="end" fill="#e6edf3" '
            f'font-family="Consolas, monospace" font-size="12" font-weight="600">{variant}</text>'
        )
        for j, (val, color) in enumerate([(a, "#58a6ff"), (b, "#a371f7"), (c, "#56d364")]):
            by = y + j * 14
            bw = max(4.0, (val / max_val) * bw_max)
            lines.append(f'<rect x="{bx0}" y="{by}" width="{bw:.1f}" height="11" rx="2" fill="{color}"/>')
            lines.append(
                f'<text x="{bx0 + bw + 6:.1f}" y="{by + 9}" fill="{color}" '
                f'font-family="Consolas, monospace" font-size="10">{val:.2f} GFLOP/s</text>'
            )

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render(ROOT / "README.md", ROOT / "docs" / "kernels.svg")
