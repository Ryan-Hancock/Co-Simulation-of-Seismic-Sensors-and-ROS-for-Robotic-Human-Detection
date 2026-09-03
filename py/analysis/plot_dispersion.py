"""Plot Rayleigh dispersion curves from geodisp, over the disba golden curve.

The golden curve is the oracle the Go solver is tested against, so drawing both
makes the agreement visible rather than a number in a test log.

    python3 py/analysis/plot_dispersion.py curves.csv -g testdata/dispersion/x.json -o d.png
"""

import argparse
import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODE_COLOURS = ["#1B5E9C", "#B3261E", "#1B5E20", "#7B3FA0"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("curves", help="CSV written by geodisp")
    ap.add_argument("-g", "--golden", help="the disba golden JSON for the same model")
    ap.add_argument("-o", "--out", default="dispersion.png")
    args = ap.parse_args()

    phase, group = defaultdict(list), defaultdict(list)
    with open(args.curves, newline="") as fh:
        for r in csv.DictReader(fh):
            m = int(r["mode"])
            phase[m].append((float(r["frequency_hz"]), float(r["phase_velocity_mps"])))
            if r["group_velocity_mps"]:
                group[m].append((float(r["frequency_hz"]), float(r["group_velocity_mps"])))

    golden, layers = {}, None
    if args.golden:
        doc = json.loads(open(args.golden).read())
        layers = doc["layers"]
        for m, c in doc["modes"].items():
            golden[int(m)] = (np.array(c["frequency_hz"]), np.array(c["phase_velocity_mps"]))

    fig, (ax, gax) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for m in sorted(phase):
        col = MODE_COLOURS[m % len(MODE_COLOURS)]
        f, c = np.array(phase[m]).T
        ax.plot(f, c, lw=1.8, color=col, label=f"mode {m}", zorder=3)
        if m in golden:
            gf, gc = golden[m]
            ax.plot(gf, gc, lw=0, marker="o", ms=3.5, mfc="none", mec=col, mew=0.9,
                    label=f"mode {m} (disba)", zorder=4)
    for m in sorted(group):
        col = MODE_COLOURS[m % len(MODE_COLOURS)]
        f, u = np.array(group[m]).T
        gax.plot(f, u, lw=1.5, color=col, ls="--", label=f"mode {m}")

    ax.set_xscale("log")
    ax.set_ylabel("phase velocity (m/s)")
    ax.set_title("Rayleigh phase velocity — lines: this solver, circles: disba")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, ncol=2)

    gax.set_xscale("log")
    gax.set_xlabel("frequency (Hz)")
    gax.set_ylabel("group velocity (m/s)")
    gax.set_title("group velocity — what actually carries the energy, and so sets arrival time")
    gax.grid(alpha=0.25, which="both")
    gax.legend(fontsize=8)

    sub = ""
    if layers:
        sub = "   ·   ".join(
            f"{l['thickness_m']:g} m Vs={l['vs_mps']:g}" if l["thickness_m"] else f"half-space Vs={l['vs_mps']:g}"
            for l in layers
        )
    fig.suptitle(f"layered Rayleigh dispersion\n{sub}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
