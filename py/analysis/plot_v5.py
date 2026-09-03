"""Plot the V5 comparison: the time-domain grid against the frequency-wavenumber path.

Two methods that share no code, drawn on the same axes. The point of the figure
is the residual panel underneath each trace — the traces themselves overlie each
other closely enough that agreement is hard to see, and disagreement easy to
miss, without it.

    python3 py/analysis/plot_v5.py v5.csv -o v5.png

Read the residual as two effects. A residual shaped like the derivative of the
trace is a timing difference, which is grid dispersion and falls linearly with
the spacing. A residual shaped like the trace itself is an amplitude difference,
which is the free surface and falls linearly too. Anything shaped like neither
is worth investigating.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID = "#1B5E9C"
FK = "#B3261E"
RESID = "#6B6B6B"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", help="CSV written by geofdtd -compare")
    ap.add_argument("-o", "--out", default="v5.png")
    args = ap.parse_args()

    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no rows in the CSV")

    cols = rows[0].keys()
    grid_cols = [c for c in cols if c.startswith("fdtd_r")]
    fk_cols = [c for c in cols if c.startswith("fk_r")]
    if not fk_cols:
        raise SystemExit("no reference columns — run geofdtd with -compare")

    t = np.array([float(r["time_s"]) for r in rows]) * 1e3
    n = len(grid_cols)
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(9, 2.6 * n), sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n, "hspace": 0.12},
    )

    for i, (gc, fc) in enumerate(zip(grid_cols, fk_cols)):
        g = np.array([float(r[gc]) for r in rows]) * 1e6
        f = np.array([float(r[fc]) for r in rows]) * 1e6
        top, bot = axes[2 * i], axes[2 * i + 1]

        top.plot(t, f, color=FK, lw=2.2, alpha=0.75, label="frequency-wavenumber")
        top.plot(t, g, color=GRID, lw=1.1, label="time-domain grid")
        top.set_ylabel("µm/s")
        rng = gc.removeprefix("fdtd_r")
        peak = np.max(np.abs(f))
        rms = np.sqrt(np.mean((g - f) ** 2) / np.mean(f**2))
        top.text(
            0.995, 0.92,
            f"r = {rng} m    peak ratio {np.max(np.abs(g)) / peak:.4f}    rms {100 * rms:.2f}%",
            transform=top.transAxes, ha="right", va="top", fontsize=9,
        )
        if i == 0:
            top.legend(loc="upper left", fontsize=9, frameon=False)
        top.grid(alpha=0.25)

        bot.plot(t, g - f, color=RESID, lw=1.0)
        bot.axhline(0, color="k", lw=0.5)
        bot.set_ylabel("residual")
        bot.grid(alpha=0.25)
        # Held at a tenth of the trace scale, so a residual that looks small
        # looks small because it is, not because the axis rescaled to it.
        bot.set_ylim(-0.1 * peak, 0.1 * peak)

    axes[-1].set_xlabel("time (ms)")
    axes[0].set_title(
        "V5: two independent solutions of the same problem\n"
        "residual panels are fixed at a tenth of the trace scale",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
