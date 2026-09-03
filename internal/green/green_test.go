package green

import (
	"math"
	"math/cmplx"
	"testing"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// The Rayleigh wave exists because a particular combination of P and SV
// potentials leaves the free surface traction-free. If the eigenfunctions are
// right, both surface tractions vanish identically; if they are wrong, the
// mode shape is not a Rayleigh wave and nothing built on it means anything.
//
// This is a first-principles check rather than a comparison against a
// published number, so it holds for every medium, not just the ones someone
// happens to have tabulated.
func TestFreeSurfaceTractionsVanish(t *testing.T) {
	for name, h := range map[string]soil.HalfSpace{
		"Poisson solid":  soil.PoissonSolid(200, 1900),
		"dry sand":       soil.DrySand(),
		"loam":           soil.Loam(),
		"weathered rock": soil.WeatheredRock(),
	} {
		t.Run(name, func(t *testing.T) {
			for _, f := range []float64{5, 20, 100} {
				e, err := NewEigenfunctions(h, 2*math.Pi*f)
				if err != nil {
					t.Fatal(err)
				}
				mu := float64(h.ShearModulus())
				lam := float64(h.LameLambda())

				// ux = i*X(z), uz = Z(z), everything carrying exp(ikx).
				xp := -e.Q*e.xq - e.S*e.xs // X'(0)
				zp := -e.Q*e.zq - e.S*e.zs // Z'(0)

				// sigma_xz = mu * (dux/dz + duz/dx) = i*mu*(X'(0) + k*Z(0))
				shear := xp + e.K*e.Vertical(0)
				if rel := math.Abs(shear) / (math.Abs(xp) + math.Abs(e.K*e.Vertical(0))); rel > 1e-12 {
					t.Errorf("%g Hz: shear traction residual %g, relative %g", f, shear, rel)
				}

				// sigma_zz = lambda*(dux/dx + duz/dz) + 2*mu*duz/dz,
				// with dux/dx = ik*(i*X) = -k*X.
				normal := lam*(-e.K*e.Horizontal(0)+zp) + 2*mu*zp
				scale := math.Abs(lam*e.K*e.Horizontal(0)) + math.Abs(2*mu*zp)
				if rel := math.Abs(normal) / scale; rel > 1e-12 {
					t.Errorf("%g Hz: normal traction residual %g, relative %g", f, normal, rel)
				}
			}
		})
	}
}

// Surface ellipticity depends on Poisson's ratio alone — not on frequency, not
// on the scale of the medium. For a Poisson solid it is 0.68127. Both facts
// are checked: the value, and its independence from everything it should be
// independent of.
func TestEllipticityOfPoissonSolid(t *testing.T) {
	h := soil.PoissonSolid(200, 1900)
	for _, f := range []float64{1, 20, 500} {
		e, err := NewEigenfunctions(h, 2*math.Pi*f)
		if err != nil {
			t.Fatal(err)
		}
		if got := e.Ellipticity(); math.Abs(got-0.68127) > 1e-4 {
			t.Errorf("%g Hz: ellipticity %.6f, want 0.68127", f, got)
		}
	}
	// Same Poisson ratio, entirely different scale: the ratio must not move.
	big := soil.PoissonSolid(2000, 2600)
	e1, _ := NewEigenfunctions(h, 2*math.Pi*20)
	e2, _ := NewEigenfunctions(big, 2*math.Pi*20)
	if math.Abs(e1.Ellipticity()-e2.Ellipticity()) > 1e-12 {
		t.Errorf("ellipticity moved with medium scale: %g vs %g", e1.Ellipticity(), e2.Ellipticity())
	}
}

// A surface wave is trapped: both potentials must decay into the half-space,
// and the displacement must be negligible a couple of wavelengths down. If a
// decay rate came out negative the "surface" wave would grow with depth.
func TestEigenfunctionsDecayWithDepth(t *testing.T) {
	h := soil.Loam()
	e, err := NewEigenfunctions(h, 2*math.Pi*20)
	if err != nil {
		t.Fatal(err)
	}
	if e.Q <= 0 || e.S <= 0 {
		t.Fatalf("decay rates must be positive, got q=%g s=%g", e.Q, e.S)
	}
	wavelength := 2 * math.Pi / e.K
	surface := math.Abs(e.Vertical(0))
	for _, depth := range []float64{1, 2, 3} {
		z := depth * wavelength
		if ratio := math.Abs(e.Vertical(z)) / surface; ratio > math.Exp(-depth) {
			t.Errorf("at %g wavelengths depth the vertical eigenfunction is still %g of its surface value", depth, ratio)
		}
	}
}

// Geometric spreading. With attenuation divided out, the Rayleigh amplitude
// must fall exactly as 1/sqrt(r) — the signature of a wave confined to a
// surface. Body waves along a free surface fall as 1/r^2, which is why by ten
// metres essentially all that remains is Rayleigh.
func TestGeometricSpreadingIsInverseSqrtRange(t *testing.T) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	const f = 20.0
	c, err := g.PhaseVelocity(f)
	if err != nil {
		t.Fatal(err)
	}
	gam := math.Atan(1/g.Soil.Qs) / math.Pi
	alpha := 2 * math.Pi * f * math.Tan(math.Pi*gam/2) / float64(c)

	ref, err := g.VelocityResponse(10, f)
	if err != nil {
		t.Fatal(err)
	}
	refAmp := cmplx.Abs(ref) * math.Exp(alpha*10)

	for _, r := range []units.Metres{1, 2, 5, 20, 50, 100} {
		h, err := g.VelocityResponse(r, f)
		if err != nil {
			t.Fatal(err)
		}
		got := cmplx.Abs(h) * math.Exp(alpha*float64(r))
		want := refAmp * math.Sqrt(10/float64(r))
		if rel := math.Abs(got-want) / want; rel > 1e-12 {
			t.Errorf("r=%g m: de-attenuated amplitude %g, want %g (rel err %g)", r, got, want, rel)
		}
	}
}

// Attenuation must be exactly exp(-omega*r / (2*Q*c)). Checked as a ratio
// between two ranges so the excitation and spreading terms cancel out and only
// the loss remains.
func TestAttenuationFollowsQ(t *testing.T) {
	for _, qs := range []float64{10, 25, 60} {
		h := soil.Loam()
		h.Qs = qs
		g := HalfSpaceGF{Soil: h}
		const (
			f     = 30.0
			r1    = 5.0
			r2    = 25.0
			delta = r2 - r1
		)
		c, err := g.PhaseVelocity(f)
		if err != nil {
			t.Fatal(err)
		}
		a, _ := g.VelocityResponse(r1, f)
		b, _ := g.VelocityResponse(r2, f)

		got := cmplx.Abs(b) / cmplx.Abs(a) * math.Sqrt(r2/r1)
		// Kjartansson's exact loss per unit range is omega*tan(pi*gamma/2)/c.
		gam := math.Atan(1/qs) / math.Pi
		want := math.Exp(-2 * math.Pi * f * delta * math.Tan(math.Pi*gam/2) / float64(c))
		if rel := math.Abs(got-want) / want; rel > 1e-12 {
			t.Errorf("Qs=%g: attenuation over %g m is %g, want %g", qs, delta, got, want)
		}
		// And it must stay within a hair of the familiar exp(-omega*r/2Qc),
		// which is the same thing to first order in 1/Q. If these ever
		// diverged, one of them would be wrong.
		approx := math.Exp(-2 * math.Pi * f * delta / (2 * qs * float64(c)))
		if rel := math.Abs(want-approx) / approx; rel > 5e-3 {
			t.Errorf("Qs=%g: exact loss %g and the 1/(2Q) approximation %g differ by %g", qs, want, approx, rel)
		}
	}
	// Higher Q must always mean less loss.
	var prev float64
	for _, qs := range []float64{5, 10, 25, 60, 200} {
		h := soil.Loam()
		h.Qs = qs
		amp, err := HalfSpaceGF{Soil: h}.VelocityResponse(30, 40)
		if err != nil {
			t.Fatal(err)
		}
		if got := cmplx.Abs(amp); got <= prev {
			t.Errorf("Qs=%g gave amplitude %g, not more than the lower-Q %g", qs, got, prev)
		} else {
			prev = got
		}
	}
}

// Attenuation is paired with dispersion, as causality requires. Applying loss
// without the matching phase term is the obvious implementation and the wrong
// one: it produces energy arriving before the first arrival. That precursor is
// small, but it lands exactly where an arrival-time picker looks, so it would
// corrupt WP3's localisation while masquerading as a physical effect.
func TestPhaseVelocityDispersesWithFrequency(t *testing.T) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	cr, err := g.Soil.RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}
	// At the reference frequency the phase velocity is the elastic one.
	c0, err := g.PhaseVelocity(units.Hertz(defaultRefFreq))
	if err != nil {
		t.Fatal(err)
	}
	if rel := math.Abs(float64(c0)-float64(cr)) / float64(cr); rel > 1e-12 {
		t.Errorf("phase velocity at the reference frequency is %g, want the elastic cR %g", c0, cr)
	}
	// Constant-Q dispersion is normal: higher frequencies travel faster.
	var prev float64
	for _, f := range []units.Hertz{1, 5, 20, 50, 200} {
		c, err := g.PhaseVelocity(f)
		if err != nil {
			t.Fatal(err)
		}
		if float64(c) <= prev {
			t.Errorf("phase velocity at %g Hz is %g, not above the lower-frequency %g", f, c, prev)
		}
		prev = float64(c)
	}
	// A lossless medium must not disperse at all.
	lossless := soil.Loam()
	lossless.Qs = 1e12
	g2 := HalfSpaceGF{Soil: lossless}
	lo, _ := g2.PhaseVelocity(1)
	hi, _ := g2.PhaseVelocity(200)
	if rel := math.Abs(float64(hi)-float64(lo)) / float64(lo); rel > 1e-9 {
		t.Errorf("a lossless medium dispersed by %g relative; attenuation and dispersion are not correctly paired", rel)
	}
}

// V9. The standing guard against the acausal-precursor failure mode, stated
// on the Green's function itself rather than on a convolved trace.
//
// The distinction matters. A convolved trace's precursor is measured against a
// peak that smoothing has already reduced, so the same absolute residue can be
// made to look like anything between a millionth and a percent depending on
// how smooth the source is. The impulse response has no such freedom: it is
// what the package is responsible for, and either it is causal or it is not.
//
// Note what the arrival time comes out as: r divided by the phase velocity at
// the *highest frequency the record can carry*, not at the reference
// frequency. A constant-Q medium disperses, and its phase velocity rises
// without bound with frequency, so it has no sharp wavefront at all — the
// earliest arrival is set by the band limit rather than by the physics. That
// is a real property of constant-Q media and not an artefact to be tuned away.
func TestGreensFunctionIsCausal(t *testing.T) {
	const (
		fs = 2000.0
		n  = 1 << 15
	)
	const r units.Metres = 20
	g := HalfSpaceGF{Soil: soil.Loam()}

	coeff := make([]complex128, n/2+1)
	for k, f := range dsp.FreqBins(n, fs) {
		h, err := g.VelocityResponse(r, units.Hertz(f))
		if err != nil {
			t.Fatal(err)
		}
		coeff[k] = h
	}
	ir := dsp.IRFFT(coeff, n)

	var peak float64
	var peakAt int
	for i, v := range ir {
		if math.Abs(v) > peak {
			peak, peakAt = math.Abs(v), i
		}
	}
	if peak == 0 {
		t.Fatal("impulse response is identically zero")
	}

	// Negative time lives in the upper half of the circular record.
	for back := 1; back < n/2; back++ {
		if ratio := math.Abs(ir[n-back]) / peak; ratio > 1e-5 {
			t.Fatalf("impulse response at t = -%.4f s is %g of peak: acausal", float64(back)/fs, ratio)
		}
	}

	// The arrival is where the band limit says it should be.
	fast, err := g.PhaseVelocity(fs / 2)
	if err != nil {
		t.Fatal(err)
	}
	want := int(math.Round(float64(r) / float64(fast) * fs))
	if peakAt < want-2 || peakAt > want+2 {
		t.Errorf("arrival at sample %d (t=%.4f s), want %d (r / phase velocity at Nyquist)", peakAt, float64(peakAt)/fs, want)
	}
}

// The arrival of a real footstep lands at the reference-frequency travel time,
// because that is where the energy actually is once the source spectrum has
// weighted it. This is the statement that matters for WP3's arrival picking.
func TestArrivalTimeMatchesRayleighVelocity(t *testing.T) {
	const fs = 2000.0
	g := HalfSpaceGF{Soil: soil.Loam()}
	const lead = 400 // source starts 0.2 s in

	// Restricted to ranges where the arrival is unambiguous. Beyond about
	// 30 m in this medium the wavelet has dispersed and attenuated enough that
	// the largest peak jumps between lobes, and the apparent velocity stops
	// tracking cR smoothly. That is not a modelling error to be tuned away —
	// it is exactly the ambiguity WP3's arrival picking will have to handle on
	// real data, and it is better to have it recorded here than discovered
	// there.
	for _, r := range []units.Metres{2, 5, 10, 20, 30} {
		force := make([]units.Newtons, int(2*fs))
		for i := range 100 {
			force[lead+i] = units.Newtons(1000 * math.Sin(math.Pi*float64(i)/100))
		}
		vel, err := g.Synthesise(force, fs, r)
		if err != nil {
			t.Fatal(err)
		}

		var peak float64
		var peakAt int
		for i, v := range vel {
			if math.Abs(float64(v)) > peak {
				peak, peakAt = math.Abs(float64(v)), i
			}
		}
		travel, err := g.TravelTime(r)
		if err != nil {
			t.Fatal(err)
		}
		got := float64(peakAt-lead) / fs
		// Compared as an apparent velocity, which is range-independent and so
		// gives one tolerance that means the same thing at every range.
		apparent := float64(r) / got
		want := float64(r) / float64(travel)
		if rel := math.Abs(apparent-want) / want; rel > 0.025 {
			t.Errorf("r=%g m: arrival at %.4f s gives apparent velocity %.1f m/s, want %.1f (rel err %g)",
				r, got, apparent, want, rel)
		}

		// Nothing before the source acts. Checked from more than 20 ms back,
		// since band-limiting smears the arrival by a few samples either way
		// and at short range the arrival is only tens of samples after the
		// source — so the samples immediately before it carry that smearing
		// rather than any leakage worth catching. The sharp statement lives in
		// TestGreensFunctionIsCausal.
		for i := range lead - 40 {
			// The bound is 5e-4 rather than the 1e-5 the impulse response
			// meets, because convolution sums the residue across every sample
			// of the source while smoothing the peak: the same absolute
			// leakage becomes a larger fraction. The physics claim is the
			// impulse-response one; this is arithmetic downstream of it.
			if ratio := math.Abs(float64(vel[i])) / peak; ratio > 5e-4 {
				t.Fatalf("r=%g m: sample %d is %g of peak, %.3f s before the source acts",
					r, i, ratio, float64(lead-i)/fs)
			}
		}
	}
}

// Ground velocity from a footstep should land in the microns-per-second range
// at ten metres. This is the one check on absolute scale available without
// Lamb's problem, and it is coarse — an order of magnitude — but it would
// catch a missing factor of 2*pi, a displacement-versus-velocity confusion, or
// a modulus in the wrong place.
func TestAbsoluteAmplitudeIsPhysicallyPlausible(t *testing.T) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	h, err := g.VelocityResponse(10, 20)
	if err != nil {
		t.Fatal(err)
	}
	// A footstep peaks near 850 N for a 75 kg walker.
	vel := cmplx.Abs(h) * 850
	if vel < 1e-6 || vel > 1e-4 {
		t.Errorf("peak ground velocity at 10 m is %g m/s; published footstep measurements are microns per second", vel)
	}
}

// Velocity is displacement differentiated, so the response must carry the
// factor of i*omega: doubling frequency at fixed everything-else doubles the
// velocity response from that factor alone.
func TestVelocityResponseCarriesIOmega(t *testing.T) {
	lossless := soil.Loam()
	lossless.Qs = 1e12 // remove attenuation and dispersion so only i*omega and 1/sqrt(k) remain
	g := HalfSpaceGF{Soil: lossless}

	a, _ := g.VelocityResponse(10, 20)
	b, _ := g.VelocityResponse(10, 40)
	// Three factors of frequency compose here, and the exponent is a useful
	// check that all three are present. The surface-wave excitation
	// uz(0)^2/I1 scales as omega, the Hankel spreading 1/sqrt(k) as
	// omega^-1/2, and the displacement-to-velocity derivative as omega:
	// omega^(3/2) overall, so 2^1.5 per octave.
	if got := cmplx.Abs(b) / cmplx.Abs(a); math.Abs(got-math.Pow(2, 1.5)) > 1e-9 {
		t.Errorf("octave amplitude ratio %g, want 2^1.5 = %g", got, math.Pow(2, 1.5))
	}
	if h, err := g.VelocityResponse(10, 0); err != nil || h != 0 {
		t.Errorf("DC response = %v (err %v), want exactly 0: a velocity transducer cannot see a static load", h, err)
	}
}

// Stiffer ground moves less under the same force.
func TestStifferSoilGivesSmallerResponse(t *testing.T) {
	var prev float64 = math.Inf(1)
	for _, h := range []soil.HalfSpace{soil.DrySand(), soil.Loam(), soil.FirmSoil(), soil.WeatheredRock()} {
		lossless := h
		lossless.Qs = 1e12
		amp, err := HalfSpaceGF{Soil: lossless}.VelocityResponse(10, 20)
		if err != nil {
			t.Fatal(err)
		}
		got := cmplx.Abs(amp)
		if got >= prev {
			t.Errorf("%s responded %g, not less than the softer medium's %g", h, got, prev)
		}
		prev = got
	}
}

func TestRejectsBadInput(t *testing.T) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	if _, err := g.VelocityResponse(0, 20); err == nil {
		t.Error("expected an error for zero range")
	}
	if _, err := g.VelocityResponse(-5, 20); err == nil {
		t.Error("expected an error for negative range")
	}
	if _, err := g.Synthesise([]units.Newtons{1}, 0, 10); err == nil {
		t.Error("expected an error for zero sample rate")
	}
	if _, err := g.Synthesise([]units.Newtons{1}, 1000, 0); err == nil {
		t.Error("expected an error for zero range")
	}
	bad := HalfSpaceGF{Soil: soil.HalfSpace{Vp: 100, Vs: 200, Density: 1700, Qs: 20}}
	if _, err := bad.Synthesise([]units.Newtons{1}, 1000, 10); err == nil {
		t.Error("expected an error for an unphysical medium")
	}
	if _, err := NewEigenfunctions(soil.Loam(), 0); err == nil {
		t.Error("expected an error for zero frequency")
	}
	out, err := g.Synthesise(nil, 1000, 10)
	if err != nil || out != nil {
		t.Errorf("empty input: got %v, %v; want nil, nil", out, err)
	}
}

func BenchmarkSynthesise(b *testing.B) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	force := make([]units.Newtons, 2000)
	for i := range force {
		force[i] = units.Newtons(math.Sin(float64(i) * 0.01))
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := g.Synthesise(force, 2000, 15); err != nil {
			b.Fatal(err)
		}
	}
}
