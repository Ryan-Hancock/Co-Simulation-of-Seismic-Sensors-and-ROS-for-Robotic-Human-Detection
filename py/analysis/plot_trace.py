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


def rng_min(cols):
    """Closest approach, ignoring the samples with no foot down."""
    if "range_m" not in cols:
        return None
    live = cols["range_m"][cols["range_m"] > 0]
    return float(np.min(live)) if live.size else None


def rng_label(r):
    return "unknown range" if r is None else f"{r:.1f} m"


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

    fig, ax = plt.subplots(4, 1, figsize=(11, 12))

    ax[0].plot(t, cols["force_n"], lw=0.9, color="#B3261E")
    ax[0].set_ylabel("force (N)")
    ax[0].set_title("total vertical ground reaction force — both feet, one walk past")
    if "range_m" in cols:
        # Range to the loaded foot, on its own axis: the rise and fall of the
        # trace below is this curve, not anything in the gait.
        rax = ax[0].twinx()
        rng = np.where(cols["range_m"] > 0, cols["range_m"], np.nan)
        rax.plot(t, rng, lw=1.2, color="#666666", ls="--")
        rax.set_ylabel("range to loaded foot (m)", color="#666666")
        rax.tick_params(axis="y", colors="#666666")

    ax[1].plot(t, cols["volts"] * 1e6, lw=0.6, color="#1B5E20")
    if "noise_v" in cols:
        ax[1].plot(t, cols["noise_v"] * 1e6, lw=0.4, color="#BBBBBB", label="sensor noise only")
        ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_ylabel("geophone output (µV)")
    ax[1].set_title(f"geophone output — closest approach {rng_label(rng_min(cols))}")

    # One footfall, close up. The heel-strike transient is the whole reason the
    # signal has any high-frequency content, so it is worth seeing on its own.
    peak = int(np.argmax(np.abs(cols["volts"])))
    lo, hi = max(0, peak - int(0.15 * fs)), min(len(t), peak + int(0.45 * fs))
    ax[2].plot(t[lo:hi], cols["volts"][lo:hi] * 1e6, lw=1.0, color="#1B5E20")
    ax[2].set_ylabel("geophone output (µV)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_title("one footfall, close up — the heel-strike transient and its coda")

    for a in ax[:3]:
        a.grid(alpha=0.25)
        a.margins(x=0)

    f, fspec = spectrum(cols["force_n"], fs)
    _, vspec = spectrum(cols["volts"], fs)
    ax[3].loglog(f[1:], fspec[1:] / np.max(fspec[1:]), lw=1.0, color="#B3261E", label="force")
    ax[3].loglog(f[1:], vspec[1:] / np.max(vspec[1:]), lw=1.0, color="#1B5E20", label="geophone output")
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
    fig.suptitle(f"geosynth — walk past a geophone\n{subtitle}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
