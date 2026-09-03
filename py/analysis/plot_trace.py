"""Plot a geosynth trace: force, ground velocity, geophone output, spectrum.

The Go side owns the runtime path and the physics; Python owns analysis and
figures. The two meet at a file, never at a function call — see the language
seam rule in docs/wp1-forward-model-plan.md. This script reads a trace CSV and
its sidecar and produces the four-panel figure that makes slice 0 legible.

    python3 py/analysis/plot_trace.py trace.csv -o trace.png
"""

import argparse
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    """Read a geosynth CSV and its .json sidecar."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

    sidecar = pathlib.Path(str(path) + ".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return cols, meta


def spectrum(x, fs):
    """One-sided amplitude spectrum, Hann-windowed."""
    w = np.hanning(len(x))
    # Compensate the window's coherent gain so amplitudes stay comparable.
    scale = 2 / w.sum()
    return np.fft.rfftfreq(len(x), 1 / fs), np.abs(np.fft.rfft(x * w)) * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="CSV written by geosynth")
    ap.add_argument("-o", "--out", default="trace.png", help="output image")
    args = ap.parse_args()

    cols, meta = load(args.trace)
    t = cols["t_s"]
    fs = 1 / (t[1] - t[0])

    res = meta.get("resolved", {})
    soil = res.get("soil", {})
    geom = res.get("geometry", {})
    rng = geom.get("range_m", float("nan"))

    fig, ax = plt.subplots(4, 1, figsize=(10, 11))

    ax[0].plot(t, cols["force_n"], lw=1.2, color="#B3261E")
    ax[0].set_ylabel("force (N)")
    ax[0].set_title("ground reaction force — one stance, vertical component")

    ax[1].plot(t, cols["velocity_mps"] * 1e6, lw=0.9, color="#1B5E9C")
    ax[1].set_ylabel("ground velocity (µm/s)")
    ax[1].set_title(f"vertical ground velocity at {rng:g} m")

    ax[2].plot(t, cols["volts"] * 1e6, lw=0.9, color="#1B5E20")
    if "noise_v" in cols:
        # The sensor's own floor, for scale: it is far below the signal, which
        # is the point worth seeing rather than asserting.
        ax[2].plot(t, cols["noise_v"] * 1e6, lw=0.5, color="#999999", label="sensor noise only")
        ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_ylabel("geophone output (µV)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_title("geophone output")

    for a in ax[:3]:
        a.grid(alpha=0.25)
        a.margins(x=0)

    # Spectra, to show where the energy actually sits. The source is smooth, so
    # slice 0's content is low — the heel-strike transient that carries the
    # high frequencies is slice 2.
    f, fspec = spectrum(cols["force_n"], fs)
    _, vspec = spectrum(cols["velocity_mps"], fs)
    ax[3].loglog(f[1:], fspec[1:] / np.max(fspec[1:]), lw=1.0, color="#B3261E", label="force")
    ax[3].loglog(f[1:], vspec[1:] / np.max(vspec[1:]), lw=1.0, color="#1B5E9C", label="ground velocity")
    ax[3].set_xlim(0.5, fs / 2)
    ax[3].set_ylim(1e-6, 2)
    ax[3].set_xlabel("frequency (Hz)")
    ax[3].set_ylabel("normalised amplitude")
    ax[3].set_title("spectra (each normalised to its own peak)")
    ax[3].grid(alpha=0.25, which="both")
    ax[3].legend(loc="lower left", fontsize=8)

    subtitle = (
        f"Vp={soil.get('Vp', '?')} Vs={soil.get('Vs', '?')} m/s, "
        f"rho={soil.get('Density', '?')} kg/m³, Qs={soil.get('Qs', '?')}"
    )
    if "config_hash" in meta:
        subtitle += f"   ·   config {meta['config_hash'][:16]}"
    fig.suptitle(f"geosynth — slice 0\n{subtitle}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
