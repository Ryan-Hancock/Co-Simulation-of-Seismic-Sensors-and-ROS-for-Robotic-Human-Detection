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

**Status: dispersion done; excitation and f–k integration outstanding.**

Done: `internal/layer` (layered medium), `internal/propmat` (Dunkin minor
propagator, secular function), `internal/disp` (bracketed root finding, mode
tracking, group velocity), `cmd/geodisp`, and `py/oracles/dispersion.py`
generating golden curves into `testdata/dispersion/`.

**V3 satisfied.** The fundamental and first two higher modes match disba to
**1e-6 relative** across four models over 2–120 Hz — homogeneous, soft-over-
stiff, a three-layer site, and a low-velocity layer trapped between stiffer
ones. The homogeneous case also reproduces slice 0's closed-form Rayleigh
velocity, and disba independently agrees, so three separate routes now concur.

**V4 satisfied, as a demonstration rather than an assertion.** `SecularNaive`
propagates the solution directly the way Thomson–Haskell does, and the two are
compared. They agree in sign while both are viable; by f·h = 3000 the direct
determinant has collapsed to 1e-12 and by 6000 to roundoff *with the wrong
sign*, while the minor formulation still returns values of order 0.1.

**Finding: the first propagator was wrong and low-frequency tests did not show
it.** In physical units the motion-stress vector mixes displacements of order
1 m with stresses of order 10⁸ Pa, so the system matrix spanned sixteen orders
of magnitude. `det P` came out at 1.003 where a traceless generator guarantees
exactly 1, and `P(d1)P(d2)` differed from `P(d1+d2)` by a factor of two. None
of that showed below ~10 Hz, because a thin layer's propagator is near the
identity and hides its own conditioning — the curves looked right until 20 Hz,
where the solver began reporting several hundred modes. Non-dimensionalising
against a reference modulus and the wavenumber fixed it: semigroup error 1.0 →
5e-15, det to 1 part in 3e9. **The lesson generalises: validate a propagator on
identities it must satisfy (det, semigroup), not only on outputs that look
right.**

**Finding: numerical eigendecomposition is unsafe for the boundary condition.**
Taking the half-space's decaying subspace from `mat.Eigen` produced a secular
function that flipped sign discontinuously where nothing physical happens — the
solver may return the two eigenvectors in either order and with either sign,
and the 2×2 minor inherits that. It looked exactly like an extra root. They are
now derived in closed form from the evanescent potentials, and vary
continuously by construction.

**f–k integration done** (`internal/fk`). Integrating over horizontal
wavenumber rather than summing residues gives the whole field at once — near
field, body waves, every mode, the static term — with no decision about which
contributions to keep. Complex arithmetic throughout, because attenuation is
what moves the Rayleigh pole off the real axis and makes the integral
convergent: Q is not a refinement here, it is what makes the problem well posed.

**The near-field result, loam at 10 Hz (λ = 18.9 m):**

| r/λ | full f–k ÷ far-field |
|---|---|
| 0.05 | **3.09×** |
| 0.11 | 2.21× |
| 0.27 | 1.50× |
| 0.53 | 1.23× |
| 1.06 | 0.93× |
| beyond | agree on average, ±20% scatter |

The far-field model is low by up to **a factor of three** inside a tenth of a
wavelength and still 20% low at half of one. The direction matters as much as
the size: the omitted terms *add*, so a detector tuned on far-field predictions
is conservative at close range, while WP3's localisation and amplitude
inversion would be biased. The residual scatter beyond one wavelength is body
waves interfering with the Rayleigh arrival — something a Rayleigh-only model
cannot produce at all.

**V2 served, by a better route than Lamb's problem.** As k grows the response
tends to C/k, and C is the medium's static near-field coefficient — for a
half-space exactly Boussinesq's (1−ν)/(2πμ). That single number pins the source
normalisation, the Hankel convention, the motion-stress scaling and the
traction boundary condition *simultaneously*; every stray factor of 2π shows up
in it. It is right to six figures in the lossless limit, and the integrated
displacement reproduces Boussinesq at range to under 1%. **Absolute amplitude
is now pinned**, which slice 0 could not do.

**Finding: an apparent 4/3 normalisation error was not one.** Isolating the
large-k asymptote *from* the quadrature showed the response exactly right with
attenuation off; the excess was Kjartansson dispersion at 0.05 Hz (three
decades below the reference frequency) compounded with a badly truncated
oscillatory tail. Testing the asymptote separately from the integral is what
separated them — a good argument for testing the pieces of a quadrature apart
from the quadrature.

**Finding: the static tail must be subtracted, not integrated.** The integrand
tends to C·J₀(kr), whose integral converges only as √(truncation) — cutting at
kr = 40 still leaves 13% error. Subtracting the asymptote and adding back its
exact integral C/r converges in a few thousand samples and makes the static
limit exact by construction.

**Remaining in this slice:**

1. **Wire the layered/near-field Green's function into the synthesis path** —
   `internal/fk` computes the response but nothing consumes it yet; `green`
   still serves `sensing`. This is where slice 5's bank belongs, since f–k is
   far too slow for the runtime path.
2. **V9 for layered media** — causality with the layered Green's function.
3. **Layered eigenfunctions** — now optional. f–k subsumes the mode sum, so
   these are only needed if WP3 wants group arrival times per mode.

**Recorded limitation:** far into the evanescent regime the P and SV vertical
wavenumbers both approach k, the two eigenvectors nearly coincide, and the
basis becomes ill-conditioned — weathered rock at k = 200 loses about four
digits. Inherent to an eigenvector formulation.

**Noted:** group velocity is computed by finite difference on the phase curve,
which goes noisy near mode osculations where dc/df is steep. Adequate for
plotting; if WP3 needs group arrival times it should be computed from the
eigenfunctions' energy integrals instead.

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

**Status: done. V5 is green.** Extrapolated to zero grid spacing, the two paths
agree on peak amplitude to **0.1–0.25%** for a half-space and **0.3–0.4%** for a
strongly layered site, with waveform residuals of **0.07–0.21% rms** and arrival
times within **30 µs** against travel times of a hundred milliseconds. Nothing
is shared between them: one expands in wavenumber and integrates a Hankel
transform, the other steps a stencil forward in time and never forms a
wavenumber.

**The elastic comparison does not exist, and that decided the design.** With Q
at 1e9 the Rayleigh pole sits *on* the wavenumber integration path and `GridFor`
asks for 2.4e11 samples — not a tuning problem but the absence of a principal
value. So V5 has to run in a lossy medium, and both sides then have to implement
the *same* loss. Kjartansson's constant-Q law has no finite-difference
counterpart: a time-domain scheme cannot evaluate a fractional power of
frequency. Fitting a relaxation spectrum to it leaves a residual that would land
in the middle of the measurement.

`internal/visco` is the answer: a single standard linear solid, which both paths
represent exactly. Its Q varies with frequency where a soil's does not, so it is
a **comparison medium, not a soil** — `fk.Medium.Relax` is nil in production.
The relaxation is scalar, one multiplier on the whole modulus tensor, so Qp
equals Qs; independent relaxation of the two moduli costs a second memory
variable per stress component and changes nothing about whether the two paths
agree.

**The scheme is first order, not second, and that had to be measured.** The
free surface is imposed on a single row, which makes it a first-order feature of
an otherwise second-order stencil — and a Rayleigh wave lives entirely on that
surface. The discrimination is clean: a **steady surface load reproduces
Boussinesq to a quarter of a percent at every spacing tried**, so the source
normalisation, the axis condition and the free surface are each exact; it is
only the dynamic surface wave that converges linearly. V5 extrapolates two runs
to zero spacing, and extrapolating with the wrong exponent would overshoot by as
much as it corrects, so `TestConvergenceIsFirstOrder` pins it rather than
assuming it.

**Finding, and a defect in the production path: `internal/fk`'s default
wavenumber sampling is about 2% high at the top of the band, and converges only
linearly.** The integrand has a near-pole at the Rayleigh wavenumber, where a
trapezoidal rule is first-order accurate at best; halving `dk` halves the
remaining error rather than quartering it. Measured at 60 Hz and 16 m: the
default 7 501 samples give a response 1.9% above the Richardson limit, and
160 000 are needed to reach a tenth of a percent. **This is the sampling the
slice 5 bank was built with**, so bank amplitudes carry roughly that error at
the top of their band. The remedy is a quadrature that subtracts the pole's
analytic contribution rather than more samples, since more samples buy only
linear improvement — and that is a change to slice 3's solver with its own
validation burden, so it is recorded here and left for slice 6 to settle.

**Isolating the measurement from its reference mattered again**, and in exactly
the way slice 5 recorded. The first comparison plateaued at 3% rms and *stayed
there under grid refinement*, which reads like a wrong answer in the grid; it
was the reference being under-converged. With the reference converged the
residual falls cleanly by a factor of two per halving. That is the second time
in two slices that the reference, not the thing under test, was the limiting
error — enough to call it a habit rather than a coincidence.

**A perfectly matched layer absorbs waves, and a static field is not a wave.**
The Boussinesq check has to be read early or in a large domain: the layer eats
the static field it cannot absorb, and the displacement drifts downward at a
rate set by how much of that field lies inside the boundary. At four times the
furthest receiver the drift is under a percent over the settling window; at
eleven times it is unmeasurable. For wave propagation the layer measures
**-70 dB**, comfortably past the -40 dB the tests demand, including for the
Rayleigh wave that arrives travelling along it — which is what the C-PML's
frequency shift is for.

**A test whose premise was wrong, recorded because the reason is worth more than
the test was.** Far-field Rayleigh amplitude falls as one over the square root
of range, so a peak-amplitude ratio between two ranges ought to show it. It does
not: the measured exponent is -0.477 and **does not move when the grid is
halved**, so it is not discretisation. The peak of the composite waveform is not
the Rayleigh amplitude — near-field and body-wave terms interfere with it, and
the spectral amplitude at fixed frequency oscillates with range accordingly. The
1/r geometry is verified by the Boussinesq check instead, which is exact rather
than asymptotic.

**Recorded limitations.** No lateral heterogeneity and no topography — this is
L2, and axisymmetry is exact for a vertical point force on horizontal layers but
cannot be extended. Attenuation in the grid is one relaxation mechanism, so the
grid cannot reproduce constant Q over a band; production attenuation stays in
the frequency domain. The memory variables inside the absorbing layer are driven
by the unstretched strain rate, which is inconsistent at second order — measured
as harmless by the -70 dB reflection, which is the only thing it could affect.

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

**Status: format, interpolation, conformance and integration done.** Remaining
is the L3 (Devito/SPECFEM3D) import path — `py/bankfmt` can already write the
format, so what is left is a driver, not a format question.

**Interpolated lookup costs 32 ns with no allocation**, against milliseconds to
compute the same response from the wavenumber integral — four orders of
magnitude, and what moves the layered near-field physics from offline into a
chunk.

**The design decision: store the frequency response, interpolate log-magnitude
and unwrapped phase separately.** Interpolating two time-domain impulse
responses gives the average of two arrivals at different times — a waveform with
two peaks where the real one has one. Phase is very nearly linear in range and
log-magnitude nearly so, so interpolating those reproduces a single arrival at
the right time. Measured at **21× more accurate** than averaging the complex
values, which is the obvious alternative and cancels badly when two ranges are
half a wavelength apart.

**Interpolation error quantified** (100 Hz over loam, against directly stored
values):

| spacing | ×limit | amplitude | phase |
|---|---|---|---|
| 0.20 m | 0.5× | 0.5% | 0.21° |
| 0.40 m | 0.9× | 1.8% | 0.81° |
| 0.80 m | 1.8× | 5.9% | 2.9° |
| 1.60 m | 3.7× | 24% | **180°** |

Quadratic in spacing while unwrapping holds, then **catastrophic rather than
gradual**: past the limit the phase error saturates at half a turn and the
response is arbitrary, not merely inaccurate. So `CheckRangeSampling` refuses to
build such a bank. The range axis has a Nyquist condition exactly analogous to
the time axis: Δr < c/2f.

**Isolating that measurement mattered.** A first attempt compared the bank
against a directly computed response and found tens of percent error at high
frequency — but the *bank* was the more converged of the two, because it
integrates on a wavenumber grid fine enough for its longest range. Comparing a
decimated bank against the fine one it came from removes the reference from the
question entirely.

**Cross-language conformance, both directions** (§6's requirement).
`py/bankfmt` reads and writes the format; Go reads Python's fixture to 3e-8
(the float32 floor) and Python reads Go's exactly. Fixtures are committed, so
the Go tests need no Python — the seam is crossed once, at file-production time.
The payload is a deterministic pattern rather than physics, so a transposed
layout, swapped real/imaginary parts, byte-order mistake or float32/float64
confusion each produce a mismatch; against real Green's functions all four would
produce numbers that still looked like Green's functions.

**Finding: a bank's usable bandwidth is not uniform across its range grid.** At
2 m the response is still rising with frequency at 600 Hz; at 10 m it has fallen
a decade and a half and is still falling; at 20 m it stops falling around 400 Hz
and flattens at ~2e-11 — the *quadrature floor*, not the medium. Synthesising
from the flat part draws in broadband noise, which inflates a trace's energy
rather than obviously corrupting its shape. A band limit must be chosen for the
**longest** range a bank will serve, or the quadrature refined where the
response is small.

The 300 Hz bank had the opposite fault — it truncated the heel-strike transient,
giving peak ratios of 0.62 at 10 m. The two errors bracket the right answer from
either side, and neither is visible without comparing against a model that does
not share them.

**Cost scaling for bank sizing**: the band limit drives everything
quadratically, since bins scale with it and the range-spacing limit scales
inversely. 300 Hz gives 270 ranges over 1–40 m in 9.4 s and 4.2 MB; 600 Hz gives
333 ranges over 1–25 m in 20.4 s and 5.2 MB.

**Layered physics now reaches the synthesis path.** `Propagation` is an
interface, so the engine takes either the analytic far-field model or a bank and
a run can be repeated with better physics and nothing else changed. The bank
path omits the fore-aft shear deliberately: a bank holds one component, and
mixing a near-field-correct vertical response with a far-field analytic
horizontal one would be incoherent. Slice 2's sweep licenses it — varying the
shear over a factor of twenty moved a walk-past by under a percent.

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
  geofdtd/               run the grid, and V5 against the f–k path
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
  fdtd/                  2D axisymmetric viscoelastic solver, C-PML (L2)
  visco/                 the one relaxation model both paths implement exactly
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
| V5 | L1 vs L2 (Go f–k vs Go axisymmetric FDTD) | **Green.** Extrapolated to zero spacing: peak amplitude to 0.1–0.25% homogeneous, 0.3–0.4% layered; residual 0.07–0.21% rms; timing within 30 µs | Independent numerics, same physics. Runs in a single-SLS comparison medium both paths represent exactly — the elastic comparison does not exist, the Rayleigh pole sits on the integration path |
| V6 | L1/L2 vs L3 (external 3D FD) | Agreement over a flat layered domain | The external-solver bridge |
| V7 | Geophone amplitude + phase response | Datasheet curves | `geophone` transfer function |
| V8 | Johnson noise PSD | `√(4k_BTR)` | Noise generator statistics |
| V9 | Causality | No energy before first arrival | Attenuation + dispersion pairing |
| V10 | FFT/convolution correctness | Parseval; convolution vs direct sum | `dsp` |
| V11 | GRF profile | Force-plate literature: peak BW, hump timing, transient rise | `grf` |
| V12 | Chunk-boundary continuity | Chunked output ≡ one long trace, to machine precision | Streaming convolution state (§7) |

**Where each lands**: V1, V2a, V7, V8, V10 in slice 0 (done); V12 in slice 1;
V11 in slice 2 (the profile and momentum checks landed early, with the source
model); V2, V3, V4, V9 in slice 3; V5 in slice 4 (green); V6 in slice 5.

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
