"""Read and write Green's function banks — the Python side of the format.

A bank is where this project's two languages meet. Go computes them from its own
wavenumber integration and reads them in the runtime path; Python writes them
from Devito or SPECFEM3D output, and reads them back for analysis. Neither
process ever calls the other: the seam is a file, which is the invariant the
whole architecture rests on.

Two implementations of one format drift. The defence is the conformance test in
conformance.py, which writes from each side and reads from the other, over a
payload chosen so that any disagreement about layout, byte order or indexing
shows up as a mismatch rather than as plausible numbers.

    from bankfmt import read, write, Bank
"""

import json
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"GEOBANK\x00"
FORMAT_VERSION = 1
PAGE_SIZE = 4096


@dataclass
class Bank:
    """A Green's function bank: a header, and complex responses on a grid."""

    header: dict
    # data[range_index, bin] as complex, matching the Go side's range-major
    # layout of interleaved float32 pairs.
    data: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.complex64))

    @property
    def bins(self):
        return self.header["samples"] // 2 + 1

    @property
    def count(self):
        return self.header["ranges"]["count"]

    def range_at(self, i):
        r = self.header["ranges"]
        if r["count"] <= 1:
            return r["min_m"]
        return r["min_m"] + (r["max_m"] - r["min_m"]) * i / (r["count"] - 1)

    def frequency_at(self, k):
        return k * self.header["sample_rate_hz"] / self.header["samples"]


def read(path):
    """Read a bank. Raises ValueError if the file is not one, or is a version
    this code does not understand."""
    with open(path, "rb") as fh:
        raw = fh.read()

    if len(raw) < 16 or raw[:8] != MAGIC:
        raise ValueError(f"{path} is not a Green's function bank")
    version, hdr_len = struct.unpack_from("<II", raw, 8)
    if version != FORMAT_VERSION:
        raise ValueError(f"{path} is format version {version}, this code reads {FORMAT_VERSION}")
    header = json.loads(raw[16 : 16 + hdr_len])

    off = 16 + hdr_len
    if off % PAGE_SIZE:
        off += PAGE_SIZE - off % PAGE_SIZE

    count = header["ranges"]["count"]
    bins = header["samples"] // 2 + 1
    want = 2 * count * bins
    flat = np.frombuffer(raw, dtype="<f4", count=want, offset=off)
    # Interleaved (re, im) pairs, range-major.
    data = flat.reshape(count, bins, 2)
    return Bank(header=header, data=(data[..., 0] + 1j * data[..., 1]).astype(np.complex64))


def write(path, bank):
    """Write a bank, byte-for-byte as the Go side would."""
    h = dict(bank.header)
    h["format_version"] = FORMAT_VERSION
    count, bins = bank.count, bank.bins
    if bank.data.shape != (count, bins):
        raise ValueError(f"data is {bank.data.shape}, header implies {(count, bins)}")

    # Go's encoding/json emits struct fields in declaration order with no
    # spaces; the exact bytes do not matter to either reader, only that the
    # length prefix agrees with the payload offset.
    hdr = json.dumps(h, separators=(",", ":")).encode()

    out = bytearray()
    out += MAGIC
    out += struct.pack("<II", FORMAT_VERSION, len(hdr))
    out += hdr
    while len(out) % PAGE_SIZE:
        out += b"\x00"

    interleaved = np.empty((count, bins, 2), dtype="<f4")
    interleaved[..., 0] = bank.data.real
    interleaved[..., 1] = bank.data.imag
    out += interleaved.tobytes()

    with open(path, "wb") as fh:
        fh.write(bytes(out))
