"""Generate golden Rayleigh dispersion curves with disba, for the Go solver to
be tested against.

This is the language seam working the way the plan intends. disba is a mature,
independently written implementation of surface-wave dispersion; using it as a
*test oracle* gets its authority into the Go test suite without putting Python
anywhere near the runtime path. It runs once, writes JSON into testdata/, and
those files are committed. Nothing at run time depends on it ever again.

Checking the Go solver against a golden file is worth much more than checking
it against itself. The failure mode this guards against is a dispersion curve
that is smooth, plausible, monotone and wrong — which is exactly what a
propagator matrix with a sign error produces.

    py/.venv/bin/python py/oracles/dispersion.py
"""

import json
import pathlib

import numpy as np
from disba import PhaseDispersion

# disba works in km, km/s and g/cm^3; this project works in SI. Converting here
# rather than in the Go test keeps the unit change on the side of the seam that
# owns the foreign convention.
M_PER_KM = 1000.0
KGM3_PER_GCM3 = 1000.0

# Models chosen to exercise the solver rather than to be realistic everywhere.
# Each is a list of (thickness_m, vp_mps, vs_mps, density_kgm3); the last entry
# is the half-space and its thickness is ignored.
MODELS = {
    "homogeneous": {
        "why": (
            "A half-space with no layering at all. The Rayleigh velocity has an "
            "exact closed form here, so this is the check that units, ordering "
            "and conventions line up before any layering is trusted."
        ),
        "layers": [(0.0, 500.0, 200.0, 1700.0)],
    },
    "soft_over_stiff": {
        "why": (
            "The classic normally dispersive case: three metres of soft soil "
            "over stiffer ground. Long waves sample the stiff half-space and "
            "travel fast, short waves stay in the soft layer and travel slow, "
            "so phase velocity falls with frequency."
        ),
        "layers": [(3.0, 400.0, 160.0, 1600.0), (0.0, 900.0, 400.0, 2000.0)],
    },
    "three_layer_site": {
        "why": (
            "Soil over weathered rock over bedrock — the structure a real "
            "survey site usually has, and where higher modes start to matter."
        ),
        "layers": [
            (2.0, 350.0, 150.0, 1600.0),
            (8.0, 900.0, 400.0, 2000.0),
            (0.0, 2200.0, 1100.0, 2400.0),
        ],
    },
    "low_velocity_layer": {
        "why": (
            "A soft layer trapped between stiffer ones. This is the awkward "
            "case: the fundamental mode can approach a higher mode closely, "
            "and a root finder that tracks by continuity rather than by "
            "bracketing will jump between them. If the Go solver survives this "
            "it will survive an ordinary site."
        ),
        "layers": [
            (1.5, 700.0, 320.0, 1900.0),
            (3.0, 400.0, 170.0, 1650.0),
            (0.0, 1400.0, 700.0, 2200.0),
        ],
    },
}

# The band this project cares about: footstep energy runs to about 100 Hz and
# the geophone corner is at 4.5 Hz.
FREQUENCIES = np.logspace(np.log10(2.0), np.log10(120.0), 60)


def curves(layers, modes=(0, 1, 2)):
    """Phase velocity in m/s for each mode, at FREQUENCIES."""
    model = np.array(
        [
            [t / M_PER_KM, vp / M_PER_KM, vs / M_PER_KM, rho / KGM3_PER_GCM3]
            for (t, vp, vs, rho) in layers
        ]
    )
    # A single-entry model still needs a thickness disba will accept; the
    # half-space's own thickness is not used.
    model[-1, 0] = max(model[-1, 0], 1.0)
    pd = PhaseDispersion(*model.T)

    periods = 1.0 / FREQUENCIES[::-1]  # disba wants increasing period
    out = {}
    for mode in modes:
        try:
            r = pd(periods, mode=mode, wave="rayleigh")
        except Exception as exc:  # a mode may not exist over the whole band
            print(f"    mode {mode}: {exc}")
            continue
        if len(r.period) == 0:
            continue
        # Back to frequency, ascending, in SI.
        f = (1.0 / np.asarray(r.period))[::-1]
        c = (np.asarray(r.velocity) * M_PER_KM)[::-1]
        out[str(mode)] = {"frequency_hz": f.tolist(), "phase_velocity_mps": c.tolist()}
    return out


def main():
    here = pathlib.Path(__file__).resolve().parents[2]
    dest = here / "testdata" / "dispersion"
    dest.mkdir(parents=True, exist_ok=True)

    for name, spec in MODELS.items():
        print(f"{name}: {len(spec['layers'])} layer(s)")
        doc = {
            "name": name,
            "why": spec["why"],
            "source": "disba 0.7.0, PhaseDispersion, Rayleigh",
            "units": "thickness m, velocity m/s, density kg/m^3, frequency Hz",
            "layers": [
                {"thickness_m": t, "vp_mps": vp, "vs_mps": vs, "density_kgm3": rho}
                for (t, vp, vs, rho) in spec["layers"]
            ],
            "modes": curves(spec["layers"]),
        }
        for mode, c in doc["modes"].items():
            v = c["phase_velocity_mps"]
            print(f"    mode {mode}: {len(v)} points, {min(v):.1f} to {max(v):.1f} m/s")
        path = dest / f"{name}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"    wrote {path.relative_to(here)}")


if __name__ == "__main__":
    main()
