"""Sobol variance decomposition over the forward model's parameter space.

The one-at-a-time sweep finds the gradient at one point and says nothing about
interactions. This apportions the *variance* of a summary statistic among the
parameters over their whole credible ranges, which is the question O4 actually
has: an axis that carries little variance is not worth randomising over, and one
that carries a lot is where the sim-to-real gap will be.

Three steps, and a file between each:

    go run ./cmd/geosweep -axes axes.json
    python3 py/analysis/sobol.py design -a axes.json -n 512 -o design.csv
    go run ./cmd/geosweep -design design.csv -o results.csv
    python3 py/analysis/sobol.py analyse -r results.csv -o sobol

The ranges come from Go, so there is no second copy of them here, and the
evaluations happen in Go, so the language seam is crossed twice per study rather
than once per sample. A Sobol design is N(k+2) runs — several thousand — and a
per-sample boundary is what would make it unaffordable.

Before trusting the estimator on a model whose answer nobody knows, run it on
one whose answer is exact:

    python3 py/analysis/sobol.py selftest

Ishigami's function has closed-form indices, including a first-order index of
exactly zero for a variable that nonetheless carries a quarter of the variance
through an interaction. An estimator that quietly reported the total index as
the first-order one would pass every plausibility check and fail this.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import qmc

# Saltelli's cross-sampling scheme. A is one sample of the whole space, B
# another independent one, and AB_i is A with the i-th column taken from B. The
# estimators below are differences between those, which is why they converge far
# faster than estimating two variances separately and subtracting.
def saltelli(n, k, seed=0):
    """Return the design as (N(k+2), k) in the unit hypercube, blocks in order
    A, B, AB_0 ... AB_{k-1}."""
    # 2k dimensions at once, so A and B come from the same low-discrepancy
    # sequence rather than from two independent ones: it is the joint filling
    # that the estimator relies on.
    sample = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(n)
    a, b = sample[:, :k], sample[:, k:]
    blocks = [a, b]
    for i in range(k):
        ab = a.copy()
        ab[:, i] = b[:, i]
        blocks.append(ab)
    return np.vstack(blocks)


def indices(y, n, k):
    """First-order and total Sobol indices from a Saltelli design's outputs.

    S1 is Saltelli's 2010 estimator and ST is Jansen's 1999 one, which are the
    pair that behave best when an index is near zero — the case that matters,
    because "this parameter does not matter" is the conclusion a randomisation
    budget acts on.
    """
    ya = y[:n]
    yb = y[n : 2 * n]
    var = np.var(np.concatenate([ya, yb]), ddof=1)
    if var == 0:
        return np.zeros(k), np.zeros(k)
    s1 = np.empty(k)
    st = np.empty(k)
    for i in range(k):
        yab = y[(i + 2) * n : (i + 3) * n]
        s1[i] = np.mean(yb * (yab - ya)) / var
        st[i] = np.mean((ya - yab) ** 2) / (2 * var)
    return s1, st


def bootstrap(y, n, k, draws=400, seed=0):
    """Confidence intervals by resampling the rows of the design.

    Resampled by row index across every block at once, not independently per
    block: the estimators are differences between matched runs, and breaking the
    matching would measure something else.
    """
    rng = np.random.default_rng(seed)
    s1s = np.empty((draws, k))
    sts = np.empty((draws, k))
    for d in range(draws):
        idx = rng.integers(0, n, n)
        resampled = np.concatenate([y[b * n : (b + 1) * n][idx] for b in range(k + 2)])
        s1s[d], sts[d] = indices(resampled, n, k)
    return (
        np.percentile(s1s, [5, 95], axis=0),
        np.percentile(sts, [5, 95], axis=0),
    )


# Ishigami's function, the standard test: exact indices, and a third variable
# whose first-order index is zero while its total index is not.
ISHIGAMI_A, ISHIGAMI_B = 7.0, 0.1


def ishigami(x):
    return (
        np.sin(x[:, 0])
        + ISHIGAMI_A * np.sin(x[:, 1]) ** 2
        + ISHIGAMI_B * x[:, 2] ** 4 * np.sin(x[:, 0])
    )


def ishigami_exact():
    a, b = ISHIGAMI_A, ISHIGAMI_B
    v1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
    v2 = a**2 / 8
    v3 = 0.0
    v13 = 8 * b**2 * np.pi**8 / 225
    var = v1 + v2 + v13
    s1 = np.array([v1, v2, v3]) / var
    st = np.array([v1 + v13, v2, v13]) / var
    return s1, st


def selftest(n, seed):
    k = 3
    unit = saltelli(n, k, seed)
    x = unit * (2 * np.pi) - np.pi
    y = ishigami(x)
    s1, st = indices(y, n, k)
    (s1lo, s1hi), (stlo, sthi) = bootstrap(y, n, k, seed=seed)
    xs1, xst = ishigami_exact()

    print(f"Ishigami, N = {n} ({n * (k + 2)} evaluations)\n")
    print(f"{'':4} {'S1':>18} {'exact':>8}   {'ST':>18} {'exact':>8}")
    worst = 0.0
    for i in range(k):
        print(
            f"x{i + 1:<3} {s1[i]:8.4f} [{s1lo[i]:6.3f},{s1hi[i]:6.3f}] {xs1[i]:8.4f}   "
            f"{st[i]:8.4f} [{stlo[i]:6.3f},{sthi[i]:6.3f}] {xst[i]:8.4f}"
        )
        worst = max(worst, abs(s1[i] - xs1[i]), abs(st[i] - xst[i]))
    print(f"\nlargest absolute error {worst:.4f}")
    # x3 is the point of the test: no first-order effect at all, and a quarter
    # of the variance through its interaction with x1.
    if abs(s1[2]) > 0.02:
        print(f"FAIL: x3 has no first-order effect but the estimator reports {s1[2]:.4f}")
        return 1
    if abs(st[2] - xst[2]) > 0.02:
        print(f"FAIL: x3's total index is {st[2]:.4f}, exactly {xst[2]:.4f}")
        return 1
    if worst > 0.02:
        print(f"FAIL: largest error {worst:.4f} exceeds 0.02")
        return 1
    print("PASS")
    return 0


def cmd_design(args):
    axes = json.load(open(args.axes))
    if args.exclude:
        drop = set(args.exclude.split(","))
        axes = [a for a in axes if a["name"] not in drop]
    k = len(axes)
    unit = saltelli(args.n, k, args.seed)
    lo = np.array([a["lo"] for a in axes])
    hi = np.array([a["hi"] for a in axes])
    # Uniform over each stated range, which is what O4 would randomise over.
    # A log-uniform prior would be defensible for Q and the coupling resonance;
    # it would change the indices, so it is a choice to make deliberately rather
    # than a detail to leave to whoever writes the sampler.
    scaled = lo + unit * (hi - lo)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([a["name"] for a in axes])
        w.writerows(scaled)
    print(
        f"{len(scaled)} runs over {k} axes (N={args.n}) -> {args.out}",
        file=sys.stderr,
    )
    json.dump(
        {"n": args.n, "seed": args.seed, "axes": [a["name"] for a in axes]},
        open(args.out + ".meta.json", "w"),
        indent=2,
    )


METRICS = ["peak_v", "rms_v", "centroid_hz", "snr_db"]

# Amplitudes are decomposed in the log.
#
# Peak and rms voltage span three orders of magnitude across these ranges — a
# walker at 3 m over firm soil against one at 30 m over sand — so their variance
# is dominated by a handful of loud runs and the decomposition ends up
# describing those rather than the model. The log is also the quantity that
# matters: nobody cares about an absolute microvolt, they care about a factor.
# The centroid is already on a natural scale and the SNR is already logarithmic.
LOGGED = {"peak_v", "rms_v"}


def cmd_analyse(args):
    # The shape is read out of the results file rather than out of a sidecar
    # written when the design was made. A results file that only works if the
    # reader still has the right sidecar is one that will eventually be read
    # against the wrong one, and the failure would be an index table rather
    # than an error.
    rows = list(csv.DictReader(open(args.results)))
    if not rows:
        raise SystemExit("no rows")
    names = [c for c in rows[0] if c not in METRICS and c != "error"]
    k = len(names)
    if len(rows) % (k + 2):
        raise SystemExit(
            f"{len(rows)} rows is not N(k+2) for {k} axes; this is not a Saltelli design"
        )
    n = len(rows) // (k + 2)
    print(f"{len(rows)} runs over {k} axes, N={n}", file=sys.stderr)

    # A rejected row is dropped from every block at once. The estimators are
    # differences between matched runs, so a row missing from one block makes
    # the whole matched set unusable; filling it in with anything would put a
    # fabricated point into a variance decomposition.
    bad = np.zeros(n, dtype=bool)
    for j, r in enumerate(rows):
        if r["error"]:
            bad[j % n] = True
    if bad.any():
        print(f"dropping {bad.sum()} of {n} matched sets that a run rejected", file=sys.stderr)
    keep = ~bad
    kept = int(keep.sum())
    if kept < 32:
        raise SystemExit(f"only {kept} usable sets; the design is mostly outside the model's domain")

    out = []
    for metric in METRICS:
        col = np.array([float(r[metric]) for r in rows])
        if metric in LOGGED:
            floor = col[col > 0]
            if floor.size == 0:
                continue
            col = np.log10(np.maximum(col, floor.min()))
        y = np.concatenate([col[b * n : (b + 1) * n][keep] for b in range(k + 2)])
        s1, st = indices(y, kept, k)
        (s1lo, s1hi), (stlo, sthi) = bootstrap(y, kept, k, seed=args.seed)
        label = f"log10({metric})" if metric in LOGGED else metric
        print(f"\n{label}   (N={kept})")
        print(f"  {'axis':<20} {'S1':>8} {'90% CI':>16}   {'ST':>8} {'90% CI':>16}")
        order = np.argsort(-st)
        for i in order:
            print(
                f"  {names[i]:<20} {s1[i]:8.3f} [{s1lo[i]:6.3f},{s1hi[i]:6.3f}]   "
                f"{st[i]:8.3f} [{stlo[i]:6.3f},{sthi[i]:6.3f}]"
            )
            out.append(
                {
                    "metric": metric,
                    "axis": names[i],
                    "s1": s1[i],
                    "s1_lo": s1lo[i],
                    "s1_hi": s1hi[i],
                    "st": st[i],
                    "st_lo": stlo[i],
                    "st_hi": sthi[i],
                }
            )
        # Total effects summing well above one means interactions; summing to
        # one means the model is additive in these parameters. Either is a
        # finding, and it is the one line of this table that says which.
        print(f"  {'sum':<20} {s1.sum():8.3f} {'':16}   {st.sum():8.3f}")

    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out}.csv", file=sys.stderr)
    if not args.no_plot:
        plot(out, names, args.out + ".png")


def plot(rows, names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 5), sharey=True)
    y = np.arange(len(names))
    for ax, metric in zip(axs, METRICS):
        sub = {r["axis"]: r for r in rows if r["metric"] == metric}
        s1 = [sub[n]["s1"] for n in names]
        st = [sub[n]["st"] for n in names]
        err = [
            [max(0, sub[n]["st"] - sub[n]["st_lo"]) for n in names],
            [max(0, sub[n]["st_hi"] - sub[n]["st"]) for n in names],
        ]
        ax.barh(y, st, color="#B3261E", alpha=0.35, label="total")
        ax.barh(y, s1, height=0.5, color="#1B5E9C", label="first order")
        ax.errorbar(st, y, xerr=err, fmt="none", ecolor="#5A5A5A", elinewidth=1, capsize=2)
        ax.set_title(f"log10({metric})" if metric in LOGGED else metric)
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(axis="x", alpha=0.25)
    axs[0].set_yticks(y, names)
    axs[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle(
        "Sobol indices: the gap between the bars is interaction, not error",
        fontsize=12,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("design", help="write a Saltelli design for geosweep")
    d.add_argument("-a", "--axes", required=True, help="axes JSON from geosweep -axes")
    d.add_argument("-n", type=int, default=256, help="base samples; the design is N(k+2) runs")
    d.add_argument("-o", "--out", default="design.csv")
    d.add_argument("--exclude", default="", help="comma-separated axes to hold at their default")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_design)

    a = sub.add_parser("analyse", help="decompose the variance of a results file")
    a.add_argument("-r", "--results", required=True, help="results CSV from geosweep -design")
    a.add_argument("-o", "--out", default="sobol")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-plot", action="store_true")
    a.set_defaults(func=cmd_analyse)

    s = sub.add_parser("selftest", help="check the estimator against Ishigami's exact indices")
    s.add_argument("-n", type=int, default=8192)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=lambda args: sys.exit(selftest(args.n, args.seed)))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
