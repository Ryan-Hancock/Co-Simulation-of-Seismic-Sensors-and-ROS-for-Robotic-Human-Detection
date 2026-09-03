"""Plot a geosweep sensitivity sweep.

One axis at a time around the reference configuration. The honest limitation to
state is that this finds the gradient at one point and says nothing about
interactions between axes; a Sobol decomposition over the joint space is a
later job.

    python3 py/analysis/plot_sweep.py sweep.csv -o sweep.png
"""

import argparse
import csv
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    axes = OrderedDict()
    for r in rows:
        axes.setdefault((r["axis"], r["unit"]), []).append(
            (float(r["value"]), float(r["rms_v"]), float(r["centroid_hz"]), float(r["snr_db"]))
        )
    return axes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="CSV written by geosweep")
    ap.add_argument("-o", "--out", default="sweep.png")
    args = ap.parse_args()

    axes = load(args.sweep)
    n = len(axes)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, grid = plt.subplots(nrow, ncol, figsize=(11, 2.1 * nrow))
    grid = np.atleast_1d(grid).ravel()

    # Sorted by how much the signal's amplitude moves across the axis, so the
    # parameters that matter are the ones read first.
    order = sorted(axes.items(), key=lambda kv: -span(kv[1]))

    for ax, ((name, unit), pts) in zip(grid, order):
        v = np.array([p[0] for p in pts])
        rms = np.array([p[1] for p in pts])
        cen = np.array([p[2] for p in pts])

        ax.plot(v, rms / rms[0], lw=1.6, color="#1B5E9C", marker="o", ms=3, label="RMS (relative)")
        ax.set_yscale("log")
        ax.set_ylabel("RMS ×", color="#1B5E9C", fontsize=8)
        ax.tick_params(axis="y", colors="#1B5E9C", labelsize=8)
        ax.axhline(1.0, color="#CCCCCC", lw=0.8, zorder=0)

        cax = ax.twinx()
        cax.plot(v, cen, lw=1.2, color="#B3261E", ls="--", marker="s", ms=2.5)
        cax.set_ylabel("centroid (Hz)", color="#B3261E", fontsize=8)
        cax.tick_params(axis="y", colors="#B3261E", labelsize=8)

        label = f"{name} ({unit})" if unit else name
        ax.set_title(f"{label}   —   RMS ×{span(pts):.2f} across the range", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.2)
        ax.margins(x=0.02)

    for ax in grid[n:]:
        ax.axis("off")

    fig.suptitle(
        "geosweep — one axis at a time about the reference walk-past\n"
        "blue: signal RMS relative to the low end of each axis   ·   red dashed: spectral centroid",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


def span(pts):
    """Largest-to-smallest RMS ratio across an axis."""
    rms = [p[1] for p in pts if p[1] > 0]
    return max(rms) / min(rms) if rms else 1.0


if __name__ == "__main__":
    main()
