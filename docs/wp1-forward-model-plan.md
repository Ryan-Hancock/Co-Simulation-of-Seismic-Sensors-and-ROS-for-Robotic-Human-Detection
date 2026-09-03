# WP1 / O1 — Forward Seismic Model: Plan

Status: draft plan, no code written yet.
Scope: Objective O1 (geophone forward model) and Work Package WP1 (forward
model + offline Green's functions), built as vertical slices rather than
objective by objective.
Languages: Go for the runtime path, Python offline. See §1.3 for the rule.

---

## 1. Languages, and where the seam goes

### 1.1 The model hierarchy

The README proposes Devito or SPECFEM3D for WP1. Those are Python-fronted /
Fortran GPU codes. Rewriting either would be a bad trade: no CUDA backend from
Go, no stencil DSL, and months reproducing validated software.

So WP1 is structured as a **hierarchy of models**:

| Level | Model | Language | Role |
|---|---|---|---|
| L0 | Analytic homogeneous half-space (Lamb's problem) | **Go** | Slice 0's model, and the unit-test oracle forever after |
| L1 | Layered half-space, frequency–wavenumber integration | **Go** | **The production forward model** |
| L2 | 2D axisymmetric elastic FDTD | **Go** | Independent numerical reference for L1 |
| L3 | 3D elastic FD / SEM over real terrain | **Python** (Devito / SPECFEM3D) | Ground truth for topography + lateral heterogeneity |

The contribution is not "we ran a solver" — it is the **quantified error and
cost at each level**, which is what licenses the real-time claim in O2/WP2.

### 1.2 Why Go for the runtime path

- The engine sits inside a soft-real-time co-simulation loop (WP2).
  Predictable latency, no interpreter, sub-millisecond GC pauses.
- O2's contribution is *characterising* coupling error and achievable
  real-time factor. If the runtime is the jittery component, we are measuring
  the language, not the coupling scheme.
- WP5 argues seismic sensing is microwatt-cheap against a lidar's tens of
  watts. The persistent sensing path should not burn CPU arguing against its
  own thesis.
- Green's-function evaluation is embarrassingly parallel over frequency,
  wavenumber and receiver. `errgroup` over a preallocated flat buffer.
- Static binary — deploying to a UGV in WP4/WP5 is `scp`.
- **The ROS 2 layer already exists in Go**: `../ros2-framework` (Conductor).
  See §6.

### 1.3 Why Python offline — and the invariant

Python is straightforwardly better for the offline half, and we use it there
without apology: the L3 solvers, validation oracles (`disba`, Computer
Programs in Seismology), ObsPy for field data in WP4, every figure and
parameter sweep, and PyTorch for WP3.

> **Invariant: the language seam is a file boundary, never a call boundary.**
> Python writes files. Go reads them. Nothing in the runtime path crosses the
> seam per-call — no FFI, no RPC, no embedded interpreter.

Why this matters enough to be an invariant rather than a preference: a
per-chunk call into Python at 100 Hz would be a permanent latency source, and
it would contaminate the exact real-time-factor measurement O2 exists to
report. Crossing once, offline, into an mmap-able file costs nothing.

Two corollaries worth stating:

1. **If a Python result cannot be reduced to a file Go reads, it belongs in
   Go.** That is the test for where a piece of work lives.
2. **Test oracles are the clean case, not an exception.** Python generates
   golden files (dispersion curves from `disba`, analytic references), those
   files are committed to `testdata/`, and Go tests read them. The ecosystem
   arrives at test-authoring time and leaves no runtime dependency behind.

The one place this bites is the bank format (§6), which needs a reader in Go
and a writer in Python. Two implementations of one format drift. Mitigation:
the written spec is the source of truth, and a round-trip conformance test
runs in CI — Python writes, Go reads, compare; then the reverse.

---

## 2. Physical model

### 2.1 Chain

```
gait phase ──▶ GRF source model ──▶ force–time function f(t) at contact point
                                              │
                            layered soil (Vp, Vs, ρ, Q, h per layer)
                                              ▼
                              Green's function G_zv(r, ω)   [m/s per N]
                                              │  convolve
                                              ▼
                                 ground velocity v_z(t) at sensor
                                              │
                          geophone transfer + coupling + noise
                                              ▼
                                        voltage V(t)
```

Every arrow is a separate package with its own validation target.

### 2.2 Source: ground reaction force

Parametric vertical GRF, since Isaac's kinematic characters produce
non-physical contact forces (README already flags this).

- Double-hump vertical profile: peaks ≈1.1–1.3 × body weight at roughly 25%
  and 75% of stance, midstance valley ≈0.7–0.8 BW.
- Heel-strike transient superimposed on the first hump, rise time ~10–30 ms.
  **This transient carries the high-frequency energy that makes footsteps
  detectable at range** — it is the part of the model that matters most and
  the part least constrained by literature.
- Stance ≈60% of gait cycle; two feet with a double-support overlap.
- Anterior–posterior horizontal component (≈0.2 BW) included — it excites
  Rayleigh waves too and is usually dropped without justification.

Three subtleties to get right, each a common error:

1. **Only the time-varying part radiates.** Static weight radiates nothing. In
   practice the geophone's own `s²` response removes DC anyway, but the model
   should be explicit rather than accidentally correct.
2. **Two feet are two sources at different positions.** The sum of both feet's
   GRF equals `W + m·a_com`, which is much smaller than either foot's peak.
   Using one foot's GRF as the total net force is wrong; using the vector sum
   as a single point source throws away the spatial dipole. Model each contact
   as its own source with its own position and time offset.
3. The GRF model is a stated assumption with a **sensitivity analysis**, not a
   hidden one. Parameters (mass, speed, surface stiffness, footwear, transient
   rise time) become the domain-randomisation axes O4 needs.

### 2.3 Propagation: layered half-space

**Decision: full frequency–wavenumber integration, not far-field mode
summation.**

This matters for this project specifically. Far-field Rayleigh asymptotics
(`1/√r` spreading, single pole residue) require `r ≫ λ`. At 20 Hz over soil
with `c_R ≈ 150 m/s`, `λ ≈ 7.5 m`. The ranges this project cares about —
a robot-mounted sensor detecting a person at 1–30 m — sit *inside* one
wavelength at the low end. Near-field terms and body-wave arrivals are not
negligible there, and a surface-wave-only model would be quietly wrong in
exactly the regime O3 operates in.

So the L1 core is:

- Propagator matrices for the layered stack, in the **Dunkin minor /
  delta-matrix** formulation. The plain Thomson–Haskell recursion loses
  precision catastrophically at high `f·h`; the minor formulation is the
  standard fix and is non-optional.
- Response evaluated on a wavenumber grid per frequency, integrated
  numerically (discrete-wavenumber / Bouchon-style, with the periodic-source
  spacing chosen to push wraparound outside the time window).
- Anelastic attenuation via complex velocities, `c(ω)(1 + i/2Q)`, with a
  logarithmic dispersion correction so the model is causal (Futterman /
  Kjartansson). Applying `exp(-ωr/2Qc)` without the paired phase term produces
  a non-causal waveform — a subtle bug that only shows up as an acausal
  precursor before the first arrival.
- Free-surface source and receiver (both at z=0) is the common case and gets a
  fast path; buried receiver supported for WP4 comparisons.

Mode summation is kept as an **optional fast approximation for large `r`**,
with the crossover range determined empirically against the full integration.
That crossover is itself a reportable WP1 result.

Emergent behaviour we should *not* hard-code, because it falls out of L1:
geometric spreading (`1/√r` Rayleigh, `1/r²` body waves at the free surface),
dispersion in layered media, frequency-dependent attenuation. The README lists
these as requirements; in this formulation they are consequences.

### 2.4 Sensor: geophone

Moving-coil velocity transducer, second-order high-pass:

```
V(s) / v_ground(s) = G · s² / (s² + 2ζω₀s + ω₀²)
```

- `G` sensitivity [V/(m/s)], `f₀` natural frequency, `ζ` total damping =
  open-circuit damping + shunt damping from the load resistor. Damping is
  load-dependent — the shunt resistor is part of the sensor model, not the
  DAQ.
- Reference part: SM-24 class. All constants read from a datasheet-backed
  config file, never hard-coded, so the model can be retargeted in WP4.
- **Ground coupling** as an explicit stage: a spike or baseplate on soil is a
  mass-on-compliance resonance, typically tens of Hz, and it is *not* a fixed
  parameter — it varies with soil, mounting, and load. This is precisely the
  "unknown and varying ground coupling" O3 has to handle, so it must be a
  first-class, parameterised, randomisable block rather than an afterthought.
- **Noise floor**, two independent contributions:
  - Johnson noise of the coil, `e_n = √(4k_BTR)`. For R=375 Ω at 300 K this is
    ≈2.5 nV/√Hz, i.e. ≈0.09 nm/s/√Hz referred to ground velocity at
    28.8 V/(m/s).
  - Ambient ground motion (Peterson NLNM as a floor, plus a cultural-noise
    model for realistic sites).
  Expected finding, worth stating early because it shapes O5: the sensor's own
  electrical noise is far below ambient ground noise, so **detection range is
  set by site noise, not by the transducer**. If the numbers say otherwise
  once computed, that is a result too.

---

## 3. Build strategy: vertical slices

Objective-by-objective is how the README is *written*; it is a poor build
order. The risk in this project is concentrated in integration — chunk
boundaries, sim-time coupling, interpolation error, the Isaac↔seismic seam —
and objective-order defers all of it to the end.

So: build the thinnest end-to-end path first, then replace one stage at a
time. Every slice ends with something that runs and something that can fail.

The **"deliberately fake"** column is load-bearing. It is what keeps a slice
small, and it is the honest record of what is not yet claimed.

### Slice 0 — "it makes a noise"

*Go. The whole chain, crudely.*

| | |
|---|---|
| **Real** | Geophone transfer function + noise (cheap and exact — no reason to fake it). L0 homogeneous half-space, Lamb's problem. Analytic double-hump GRF. |
| **Fake** | One soil, hard-coded. One foot. Far-field only. No layering, no dispersion, no chunking, no ROS. |
| **Ships** | `geosynth` CLI: config in, trace file out. Python plots it. |
| **Exit** | V1, V7, V8, V10, plus the eigenfunction checks below. A waveform that looks like a footstep. |

**Status: done.** `cmd/geosynth` runs the chain from a config file;
`py/analysis/plot_trace.py` plots it. Reference run — 75 kg walker at 10 m over
loam, SM-24 shunted to 0.7 — gives 846 N peak force, 1.81 µm/s ground velocity,
46 µV, 56 dB above the sensor's own floor.

**Scope change: V2 moved to slice 3.** Lamb's problem was to be slice 0's
oracle, but it pins *absolute amplitude in the near field*, and slice 0
explicitly declares far-field only. Implementing the full Pekeris solution to
validate a model that does not claim the regime it tests would have been work
spent in the wrong slice. What replaced it is stronger than a tolerance-fudged
version of V2 would have been: the excitation is built from the half-space
Rayleigh eigenfunctions, and those are checked from first principles — both
free-surface tractions vanish to 1e-12 for every medium tested, and the surface
ellipticity comes out at 0.681250 against the textbook 0.68127. Absolute scale
is still unpinned until V2 lands in slice 3; the amplitudes above are
physically plausible against published footstep measurements but are not yet
validated.

**Kjartansson, not Futterman.** The plan said "causal complex-velocity
attenuation" without naming a model. Futterman's logarithmic pairing was tried
first and rejected on measurement: it is causal only to first order in 1/Q and
left a percent-level precursor at soil Q values, and its logarithm diverges at
zero frequency so it needs an arbitrary clamp that breaks the very
Kramers-Kronig pairing it exists to provide. Kjartansson's constant-Q model is
exactly causal, needs no clamp, and agrees with the familiar exp(-ωr/2Qc) to
better than 0.1% above Q≈20.

**Finding: the radiated field is dominated by contact onset, not by the
peaks.** Velocity response scales as ω^(3/2), so it weights the fastest-changing
part of the force. The smooth middle of a stance radiates almost nothing and
the double hump is nearly invisible in the ground motion; what shows is
heel-strike and toe-off, interfering as a spectral comb at one over the stance
duration. Two consequences. Slice 0's amplitude is dominated by the taper —
the least constrained part of the source model, fixed by the momentum
constraint rather than measured — so it should be read as more provisional than
the propagation. And slice 2's heel-strike transient will not merely add
high-frequency content, it will take over the signal.

**Finding: a constant-Q medium has no sharp wavefront.** Phase velocity rises
without bound with frequency, so the earliest arrival is set by the band limit
rather than by the physics. Causality is therefore asserted on the impulse
response (causal to 1e-5 of peak), not on a convolved trace, where the same
absolute residue can be made to look like anything between a millionth and a
percent depending on how smooth the source is.

**Finding: beyond ~30 m the arrival becomes ambiguous.** The wavelet has
dispersed and attenuated enough that the largest peak jumps between lobes and
apparent velocity stops tracking cR smoothly. Not an artefact to tune away —
it is the ambiguity WP3's arrival picking will meet on real data.

### Slice 1 — "it's on the graph"

*Go + Conductor. Integration risk, early.*

| | |
|---|---|
| **Real** | `GeophoneChunk.msg` via `conductor msggen`. Conductor node publishing under `/clock`. Streaming convolution state across chunk boundaries. |
| **Fake** | Still slice 0's physics. Source can be a canned replay. |
| **Ships** | A node on a live ROS 2 graph emitting chunks in simulated time. |
| **Exit** | V12 (chunk continuity), transport benchmarked at chunk rate under sim time. |

**Status: done.** `cmd/geonode` publishes `geosim_msgs/GeophoneChunk` on a
Conductor node under `/clock`; `conductor check` validates the graph with no
warnings. V12 holds twice over: inside the convolver to machine precision, and
end-to-end — the published stream reassembled and compared against the offline
model synthesised in one piece, agreeing to float32 precision across the
engine's buffering, the timer, encoding, and the transport.

**Real-time factor, the number that gates WP2** (one receiver, loam, 2 kHz,
zero allocations per chunk):

| chunk | per chunk | share of real time |
|---|---|---|
| 5 ms | 8.7 µs | 0.17% |
| 10 ms | 7.5 µs | 0.075% |
| 100 ms | 18.0 µs | 0.018% |

Cost is flat across a 20× range of chunk sizes, which was the point of using
partitioned rather than plain overlap-add: chunk length is the knob O2 sweeps,
and a synthesis cost that blew up as it turned would make coupling error and
compute cost inseparable in the results.

**Finding: `Synthesise` was carrying an acausal artefact, and the streaming
path exposed it.** The two paths disagreed by ~1% of peak in the coda. It was
not truncation — the discarded impulse-response tail holds 3.5e-14 of the
energy, and the disagreement was independent of response length, transform
size, and the anti-alias taper. It was the acausal pre-ring of the band limit,
which a frequency-domain product keeps and a causal impulse response discards.
Confirmed by adding the negative-time taps back, which reproduced the spectral
answer to 3e-14. Both paths now run through the same causal impulse response,
so the offline path lost an artefact it should never have had.

**Finding: publishing a reused buffer is silently wrong on an in-process
transport.** Messages pass by reference, so every subscriber held the same
array and saw only the most recent chunk — each chunk individually correct,
only the assembled stream wrong. Caught by the reassembly test, invisible to
inspection.

**Open, and an O2 question rather than a defect**: the chunk stamp is the
runtime's clock reading at the tick, while sample times within a chunk come
from the sample rate. How far those drift apart under a coarse `/clock` is
exactly the interface-error question O2 exists to characterise.

### Slice 2 — "the source is defensible"

*Go, with Python sensitivity sweeps.*

| | |
|---|---|
| **Real** | Full GRF model per §2.2 — double hump, heel-strike transient, two feet as separate co-located-in-time sources, AP horizontal component, gait phase. Radiating (time-varying) part only. |
| **Fake** | Propagation still L0. |
| **Ships** | A gait that produces a plausible footstep train. |
| **Exit** | V11. Sensitivity sweep over mass, speed, transient rise time. |

**Status: done.** `cmd/geosweep` runs the sweep; `py/analysis/plot_sweep.py`
plots it. Reference walk-past — 75 kg at 1.3 m/s passing a sensor at 10 m over
loam — gives 30 footfalls over 15.4 s, peak 226 µV, 70 dB above the sensor
floor, against slice 0's 46 µV for a single stance at the same range. Output
now carries to about 300 Hz where slice 0 had died by 50.

The GRF is the weakest link in the chain (the README says so). Making it a
slice of its own, with the sensitivity analysis attached, is what turns it
from a hidden assumption into a stated one.

**Sensitivity, ranked by how far the signal moves:**

| axis | range | RMS | centroid |
|---|---|---|---|
| shear velocity | 120–400 m/s | ×15.0 | 64 → 109 Hz |
| range | 3–30 m | ×4.1 | 109 → 47 Hz |
| transient peak | 0.05–0.7 BW | ×3.2 | 53 → 88 Hz |
| body mass | 50–120 kg | ×2.4 | flat |
| transient rise | 6–30 ms | ×2.4 | 92 → 70 Hz |
| soil Q | 8–60 | ×1.8 | 44 → 111 Hz |
| coupling resonance | 15–250 Hz | ×1.7 | 23 → 91 Hz |
| stance duration | 0.45–0.8 s | ×1.2 | |
| walking speed | 0.8–1.8 m/s | ×1.0 | |
| fore-aft shear | 0.02–0.4 BW | ×1.0 | |

**Soil shear velocity dominates by a factor of four.** It is the parameter WP4
most needs to measure rather than assume, which argues for a refraction survey
alongside every field recording (§9.1).

**The coupling resonance curve makes the O3 warning quantitative**: below about
45 Hz it sits inside the band and costs up to 40% of the signal; above it the
sensor is transparent. A robot-mounted geophone lives near that knee.

**Finding: the fore-aft shear contributes essentially nothing**, for two
compounding reasons. It radiates with a cos(azimuth) pattern that nulls exactly
where a walk-past is loudest, and it is a smooth half-cycle over the whole
stance with no impact transient, so its energy sits at a few hertz where
propagation and the geophone are both weak. Even head-on it moves the trace by
a fraction of a percent. Implementing it was not wasted — adding it
isotropically would have overstated a walk-past and misjudged the in-line case
the other way — but the vertical force is the whole story here.

**Limitation: walking speed barely matters, and that is about the model, not
about walking.** Speed lengthens the stride but leaves cadence alone, because
the gait cycle derives from stance duration and stance duration is an
independent input; and per-step force is speed-independent. Real walkers do
neither — cadence rises with speed, stance shortens, peak force climbs from
~1.1 BW at a stroll to over 1.3 brisk. Sweeping speed meaningfully means
co-varying all three, which needs force-plate relations WP4 can supply. **This
is the first thing to fix in the source model.**

**Finding: the taper was swallowing the transient.** The smooth curve's taper
runs ~60 ms, three times the transient's rise, and taking the transient through
it suppressed its peak fourfold. The force trace still looked entirely like a
footstep; only the radiated spectrum would have shown the loss, and by then it
would have read as a propagation problem.

**Finding: a walk has to be able to end.** An unbounded `Walk` had the engine
discovering footfalls at ever-increasing range forever and building a Green's
function for each *inside the real-time loop* — 668 µs per chunk against a
10 ms budget. Nothing was wrong with the waveform; only the benchmark found it.

**Steady-state cost** (one receiver, 2 kHz, 10 ms chunk, 5 voices live):
**105 µs per chunk, 1.05% of real time**, against slice 1's 0.075% for a single
fixed source. Each live contact costs two convolutions — vertical force and the
radial shear component — and outlives its own stance by the coda still
arriving.

**Range quantisation**: impulse responses are cached by range at 0.02 m. The
dominant error is not amplitude, which goes as 1/√r and barely moves, but
*arrival time* — and WP3 localises from arrival times. 0.02 m is ~0.1 ms, about
2 cm of localisation error; the 0.1 m first tried would have been 10 cm. Slice
5's bank replaces this with proper interpolation.

### Slice 3 — "right at the ranges we actually care about"

*Go core, Python oracle. The hard one.*

| | |
|---|---|
| **Real** | L1: layered half-space, Dunkin minor propagator matrices, full f–k integration, causal complex-velocity Q. Near-field and body-wave terms. |
| **Fake** | Laterally homogeneous. Flat surface. |
| **Ships** | Dispersion curves, and waveforms valid at 1–30 m. |
| **Exit** | V3, V4, V9. Golden dispersion files from `disba`/CPS committed to `testdata/`. |

This is the slice that earns "physically grounded" in O1. It is also the one
most likely to overrun — see the root-finding and near-field risks in §7.

### Slice 4 — "independently verified"

*Go.*

| | |
|---|---|
| **Real** | L2 2D axisymmetric elastic FDTD, C-PML boundaries. |
| **Fake** | Still no lateral heterogeneity or topography. |
| **Ships** | A second, independent numerical route to the same answer. |
| **Exit** | V5 — L1 and L2 agree to a stated tolerance. |

V2 says the analytic path is right; V5 says the layered path is right by a
route that shares no code with it. These two are the load-bearing validations.

### Slice 5 — "fast enough, and it scales"

*Go runtime, Python solver driver.*

| | |
|---|---|
| **Real** | Bank format + spec + conformance test. Interpolation. L3 import path from Devito/SPECFEM3D output. Benchmarks. |
| **Fake** | L3 runs stay small; full terrain is WP2's problem. |
| **Ships** | The real-time-factor number that gates WP2's architecture. |
| **Exit** | V6, interpolation error quantified, throughput measured. |

**The number that matters**: wall-clock to synthesise 1 s of 2 kHz trace for N
receivers. If that is not comfortably faster than real time with margin for
Isaac and ROS 2 on the same machine, WP2's architecture has to change. Better
to learn it here than three work packages later.

### Slice 6 — "characterised"

*Python.*

| | |
|---|---|
| **Real** | Full sensitivity analysis (Sobol over soil + subject + coupling). The hierarchy error/cost table. Validation report. |
| **Ships** | O1 acceptance, and the priors for O4's domain-randomisation ranges. |
| **Exit** | V1–V12 green in CI; report regenerable from one command. |

### What each slice buys

Ordering is not arbitrary. Slices 0–2 are cheap and de-risk integration;
slice 3 is where the research sits; slices 4–6 are what make the claims
defensible. If the project is squeezed, 4 and 6 compress and 3 does not.

---

## 4. Repo layout

Go module path: `github.com/<user>/co-sim-geophone-robot` — placeholder, needs
setting before first commit.

```
cmd/                     Go CLIs
  geosynth/              synthesise a trace from a config
  geodisp/               dispersion curves + eigenfunctions
  geobank/               build / inspect / validate GF banks
  geonode/               the Conductor node (slice 1 onward)

internal/                Go, runtime path
  units/                 SI discipline: named types at API edges
  soil/                  layered medium, presets, randomisation ranges
  propmat/               Dunkin minor propagator matrices, secular function
  disp/                  Rayleigh root finding, mode tracking, eigenfunctions
  green/                 f–k integration, GF assembly, interpolation
  bank/                  bank reader (mmap), format v1
  grf/                   gait phase, GRF profiles, multi-contact scheduling
  geophone/              transfer function, coupling model, noise
  dsp/                   FFT wrapper, streaming convolution, filters
  fdtd/                  2D axisymmetric elastic solver (L2)
  trace/                 waveform type + metadata, SAC/miniSEED export
  config/                schema, hashing, provenance

py/                      Python, offline only
  oracles/               generate golden testdata (disba, CPS, analytic)
  solvers/               Devito / SPECFEM3D drivers → bank format
  bankfmt/               bank writer + reference reader
  analysis/              figures, sweeps, the validation report

msgs/                    .msg definitions → conductor msggen
testdata/                golden files, committed
docs/
```

### Go design decisions

- **`float64` in the physics core, `float32` on disk.** Banks are large (§6);
  the propagator recursion is not float32-safe.
- **One dependency that matters: gonum.** `dsp/fourier` for FFT (stdlib has
  none), plus matrices, root finders, interpolation. Resist adding more.
- **Units discipline is a real hazard here** — N, m/s, V, m/s², and
  displacement vs velocity vs acceleration all in one pipeline. Named types
  at package boundaries, SI internally, conversion only at I/O.
- **Reproducibility is a hard requirement**, since O4's domain randomisation
  is meaningless if runs do not replay exactly. Every trace carries the hash
  of its resolved config and the explicit RNG seed.
- **Chunk-oriented synthesis API** (§7): the engine fills a caller-supplied
  `[]float32` window, carrying convolution state across chunk boundaries.
  Hot path allocation-free — verify with `-benchmem` showing 0 allocs/op.
- **Concurrency**: parallel over frequency in `green`, over receiver at
  runtime.

### Python discipline

Offline does not mean unversioned. `py/` gets pinned dependencies, is
importable as a package, and its outputs carry the same config hash as the Go
side so a figure can be traced to the run that produced it.

---

## 5. Validation — the actual deliverable

O1 says "physically grounded". That claim is only worth anything if it is
tested against things with known answers. Each is a Go test, run in CI.

| # | Target | Known value / source | Tests |
|---|---|---|---|
| V1 | Rayleigh velocity, homogeneous Poisson solid | `c_R/β = 0.9194` at ν=0.25 | Secular function + root finder |
| V2 | Lamb's problem: vertical surface response to vertical surface point force | Closed-form (Pekeris / Mooney) | Absolute amplitude and the near field. **Moved to slice 3** — it tests a regime slice 0 does not claim |
| V2a | Rayleigh eigenfunctions | Free-surface tractions vanish; ellipticity 0.68127 for a Poisson solid | The excitation, from first principles (slice 0's replacement for V2) |
| V3 | Layered dispersion curves | Published models; cross-check vs Computer Programs in Seismology / `disba` | `propmat` + `disp` incl. higher modes |
| V4 | High `f·h` stability | Thomson–Haskell diverges, Dunkin minors do not | The minor formulation specifically |
| V5 | L1 vs L2 (Go f–k vs Go axisymmetric FDTD) | Agreement to stated tolerance | Independent numerics, same physics |
| V6 | L1/L2 vs L3 (external 3D FD) | Agreement over a flat layered domain | The external-solver bridge |
| V7 | Geophone amplitude + phase response | Datasheet curves | `geophone` transfer function |
| V8 | Johnson noise PSD | `√(4k_BTR)` | Noise generator statistics |
| V9 | Causality | No energy before first arrival | Attenuation + dispersion pairing |
| V10 | FFT/convolution correctness | Parseval; convolution vs direct sum | `dsp` |
| V11 | GRF profile | Force-plate literature: peak BW, hump timing, transient rise | `grf` |
| V12 | Chunk-boundary continuity | Chunked output ≡ one long trace, to machine precision | Streaming convolution state (§7) |

**Where each lands**: V1, V2a, V7, V8, V10 in slice 0 (done); V12 in slice 1;
V11 in slice 2 (the profile and momentum checks landed early, with the source
model); V2, V3, V4, V9 in slice 3; V5 in slice 4; V6 in slice 5.

V2 and V5 are the load-bearing ones. V2 says the analytic path is right; V5
says the layered path is right by independent numerical route. Everything else
is a guard rail.

**Sensitivity analysis** (explicitly required by the README for the GRF, and
worth extending to soil): one-at-a-time and Sobol-style sweeps over body mass,
gait speed, transient rise time, `Vs`, `Q`, layer depth, coupling stiffness →
report effect on peak amplitude, spectral centroid, and SNR-vs-range slope.
This output doubles as the prior for O4's domain-randomisation ranges.

---

## 6. Green's-function banks — the seam in practice

The README's "grid of source positions × grid of receiver positions" needs
care, because the naive version does not fit on disk.

- **Laterally homogeneous medium (the L1 baseline): the problem collapses.**
  The GF depends only on `|r_src − r_rcv|`, so one dispersion computation and
  one range axis cover *every* source–receiver pair. Storage is kilobytes,
  runtime is an interpolation plus a convolution. This is what makes real-time
  feasible, and it is worth saying plainly rather than tabulating 3D banks out
  of habit.
- **Reciprocity** halves the work when the medium is *not* laterally
  homogeneous: one simulation per *receiver*, recording at all source
  positions — `N_rcv` runs, not `N_src × N_rcv`.
- **Only L3 needs a tabulated 3D bank**, for lateral heterogeneity and
  topography. Scale check: 2 s at 2 kHz float32 = 16 kB per pair; a 100×100
  source grid × 50 receivers ≈ 8 GB. Tractable, but it needs a chunked,
  memory-mappable format with an index — not one file per pair.

### Format v1

This file *is* the language seam, so it gets a written spec before either
implementation:

- Self-describing header: soil model, grids, sampling, solver provenance,
  config hash, format version.
- Chunked `float32` payload, `mmap`-friendly, index at a known offset.
- Versioned from v1 so WP2 does not break on a mid-project format change.
- **Round-trip conformance test in CI**: `py/bankfmt` writes, `internal/bank`
  reads, compare; then the reverse. This is the only defence against the two
  implementations drifting, and it is cheap.

Runtime interpolation across source position, receiver position and range
introduces its own error. **Quantify it** — interpolated vs directly computed
GF — and report it next to WP2's coupling-step error, since the two combine
into the total error budget for the co-simulation.

---

## 7. Interface to WP2 — Conductor

**We already have a Go ROS 2 framework.** `../ros2-framework` ("Conductor")
is a declarative Go application framework that speaks ROS 2 over Zenoh. The
runtime path needs no FFI and no Python.

Worth being explicit, since §1.3 puts Python in the project: **this is not a
language choice that has to be made once.** Isaac Sim is Python-scripted, so
the Isaac side of WP2 will be `rclpy` regardless. Conductor nodes and rclpy
nodes coexist on the same graph — that is what rmw_zenoh wire compatibility
buys, and what the interop suite proves. ROS 2 is the boundary; it is doing
its job.

What Conductor already provides that WP1/WP2 need:

| Capability | Status | Why it matters here |
|---|---|---|
| `transport/rmwzenoh` | Wire formats verified against live rmw_zenoh 0.10 (ROS 2 Lyrical), golden-value tests | A conductor process is indistinguishable from a ROS 2 node on the graph — Isaac Sim talks to it natively |
| `/clock` simulated time | `simClock`, honours `use_sim_time`; timers, stamps and mission timeouts all follow it | **Directly the O2 synchronisation substrate.** Waiters anchor at clock start rather than the zero time, and time going backwards (sim reset) is handled |
| `conductor msggen` | Generates Go types from `.msg`/`.srv`/`.action`, computes REP-2011 RIHS01 type hashes | Custom geophone messages are a solved problem, with hashes that match real ROS peers |
| CDR codec | Slices and fixed arrays supported | `float32[]` waveform chunks serialise correctly |
| Static graph checking | `conductor check` validates topics, types and QoS at build time | Co-simulation topology becomes a compile-time artifact — a genuinely reportable property for a *co-simulation framework* contribution, not just convenience |
| `make interop` | Test matrix against real ROS 2 with a live router | The sim-to-ROS bridge is regression-tested, not assumed |

### What this changes for WP1

1. **No FFI constraint on the engine API.** The synthesis engine is an
   ordinary Go package that a Conductor node imports. Drop the flat-buffer
   discipline imposed for a C boundary; keep the allocation-free hot path,
   which is still required for soft real time.

2. **Chunked output is the design point.** A geophone samples at 1–2 kHz. One
   ROS message per sample is the wrong shape — 2000 msgs/s per sensor through
   a node executor is avoidable waste. Publish **chunks**: a
   `GeophoneChunk.msg` carrying a header, `float32[] samples`, sample rate,
   and the sim-time stamp of sample zero. A 10 ms chunk is 20 samples at
   2 kHz, i.e. 100 msgs/s per sensor.

   This is a WP1 decision, not a WP2 one, because it dictates the engine
   signature: `Synthesise(dst []float32, t0, dt, nSamples)` filling a window,
   with the convolution state carried across chunk boundaries. Getting that
   boundary condition right — no discontinuity at chunk edges from a
   truncated convolution tail — is V12, tested in slice 1.

3. **Chunk length is an O2 research parameter, not a constant.** It sets the
   coupling granularity between Isaac (60–240 Hz) and the seismic stream
   (1–2 kHz), so it is exactly the knob the "step-size selection and
   extrapolation error at the interface" question in O2 is about. Make it
   configurable from the start and sweep it.

4. **Message definitions are a WP1 deliverable.** Define the `.msg` set now —
   `GeophoneChunk`, `FootstepEvent`, `SoilModel`, `ContactForce` — and
   generate with `conductor msggen`. `ContactForce` should be general enough
   to carry robot wheel/track contacts as well as feet, per §9.3.

### Open items

- Confirm Conductor's timer and executor hold up at chunk rates under
  simulated time — benchmarked in slice 1, not assumed. The chunking
  in (2) keeps this well away from the interesting limits, which is part of
  why it is the right design.
- Conductor is `go 1.24`; this repo can be `go 1.26`. Fine, but pin
  deliberately.
- Conductor's Zenoh transport is cgo (`-tags zenoh`) with a vendored
  `zenoh-c`. That affects cross-compilation for the WP4/WP5 UGV deployment —
  worth confirming the target build early rather than at field-trial time.

*Also noted:* `../vibration-ml` in the same parent directory may be relevant
to WP3/O3. Not examined yet; worth a look before that work package starts.

## 8. Risks

| Risk | Slice | Impact | Mitigation |
|---|---|---|---|
| Rayleigh root finder misses/jumps modes in layered stacks | 3 | Silently wrong dispersion — the worst failure mode, because it looks plausible | Dunkin minors; bracket by mode continuity in `f`; V3 against golden `disba`/CPS files |
| Near-field invalidity at short range | 3 | Wrong at 1–10 m, the range O3 needs | Full f–k integration rather than mode summation (§2.3); crossover measured, not assumed |
| Acausal waveforms from unpaired Q | 3 | Precursor energy before the first arrival; corrupts arrival-time picking in WP3 | Causal complex-velocity attenuation; V9 as a standing test |
| Chunk-edge discontinuity | 1 | Looks exactly like a real seismic transient; would survive into WP3 detection | V12 from slice 1, before any physics depends on the transport |
| GRF model is under-constrained | 2 | The weakest link in the chain (README agrees) | Stated assumption + sensitivity analysis + WP4 field check; never presented as ground truth |
| Coupling resonance treated as fixed | 0 | Sim-to-real gap in O4 that is easy to misattribute to soil | Coupling is a parameterised, randomisable block from slice 0 |
| Bank format implementations drift | 5 | Silent corruption across the language seam | Written spec is the source of truth; round-trip conformance test in CI (§6) |
| Nonlinear soil very close to the foot | 3 | Linear-elastic assumption breaks in the near field | Document the validity range; if it bites, an equivalent-linear source correction near contact |
| Float32 banks lose precision | 5 | Small amplitudes at long range | Store scaled; V6 checks against a float64 reference |
| Slice 3 overruns | 3 | It is the research, and it is the hard one | Slices 0–2 already ship something demonstrable; 4 and 6 compress if needed, 3 does not |

---

## 9. Open questions

1. **Soil parameter ranges** — which soils does WP4 realistically get access
   to? That sets the randomisation ranges and should feed back into slice 3
   rather than being guessed.
2. **Terrain coupling to Isaac** — does the L1 flat-layered model suffice for
   the WP5 scenarios (wind farm / solar site), or is topography a first-order
   effect that forces L3 into the runtime path? This decides whether §6's
   "the problem collapses" holds in practice.
3. **Is the robot's own ego-noise source model the same machinery?** It should
   be — O3 says "predict via the same Green's function machinery" — so `grf`
   needs to handle wheel/track contact forces, not just feet. **Design it as a
   general multi-contact force scheduler in slice 2**, not a footstep-specific
   one. Retrofitting this later is expensive.
4. **Module path / repo hosting** — needs fixing before the first commit.
