"""Write a bank Go can read, and check the one Go wrote.

Run from the repository root:

    py/.venv/bin/python py/bankfmt/conformance.py write testdata/bank/python_written.bank
    py/.venv/bin/python py/bankfmt/conformance.py check testdata/bank/go_written.bank

The payload is a deterministic pattern rather than real physics, chosen so that
every way the two implementations could disagree produces a mismatch instead of
plausible numbers. Range index and bin index enter it differently, so a
transposed layout fails; real and imaginary parts differ, so a swapped pair
fails; and the values span several decades, so a float32/float64 confusion
fails.
"""

import sys

import numpy as np

from bankfmt import Bank, read, write

COUNT = 7
SAMPLES = 32
RATE = 2000.0


def pattern(i, k):
    """The conformance payload: distinct in each index, and wide in magnitude."""
    return complex((i + 1) * 10.0 ** (k % 5 - 2), -(k + 1) * 0.25 + i)


def make():
    bins = SAMPLES // 2 + 1
    data = np.zeros((COUNT, bins), dtype=np.complex64)
    for i in range(COUNT):
        for k in range(bins):
            data[i, k] = pattern(i, k)
    header = {
        "format_version": 1,
        "provenance": {"solver": "python conformance fixture"},
        "medium": [{"thickness_m": 0, "vp_mps": 500, "vs_mps": 200, "density_kgm3": 1700}],
        "sample_rate_hz": RATE,
        "samples": SAMPLES,
        "ranges": {"min_m": 1.0, "max_m": 4.0, "count": COUNT},
        "component": "conformance fixture, not physical",
        "units": "m/N",
    }
    return Bank(header=header, data=data)


def check(path):
    b = read(path)
    if b.count != COUNT or b.header["samples"] != SAMPLES:
        raise SystemExit(f"FAIL: {path} has count={b.count} samples={b.header['samples']}")
    worst = 0.0
    for i in range(b.count):
        for k in range(b.bins):
            want = np.complex64(pattern(i, k))
            got = b.data[i, k]
            denom = max(abs(want), 1e-30)
            worst = max(worst, abs(got - want) / denom)
    if worst > 1e-6:
        raise SystemExit(f"FAIL: worst relative difference {worst:.3e} in {path}")
    print(f"ok: {path} matches the conformance pattern (worst relative difference {worst:.3e})")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: conformance.py write|check <path>")
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "write":
        write(path, make())
        print(f"wrote {path}")
    elif cmd == "check":
        check(path)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
