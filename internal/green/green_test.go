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
	//
	// The bound is 2e-4 rather than zero because this model is an asymptotic
	// approximation, and an asymptotic approximation is not exactly causal. The
	// far-field form is derived for kr much greater than one; at the low
	// frequencies where that fails it gets the response wrong, and a wrong
	// low-frequency response is what an acausal precursor is made of. Measured
	// at 6e-5 of peak, which is the price of the approximation rather than a
	// defect in it — the exact solution is causal, and internal/fk agrees with
	// a time-domain solver to a fraction of a percent (V5).
	//
	// This bound was 1e-5 while the model carried a ninety-degree phase error,
	// and the error made the response look *more* causal than the physics is:
	// rotating the phase smooths the onset, and a smooth onset band-limits
	// cleanly. A tighter causality bound was evidence of a bug, not of quality.
	for back := 1; back < n/2; back++ {
		if ratio := math.Abs(ir[n-back]) / peak; ratio > 2e-4 {
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

// The Rayleigh wavetrain moves out at the Rayleigh velocity, measured as a lag
// between ranges rather than as an absolute arrival time.
//
// This test used to assert that the largest peak of a footfall response arrives
// at r/cR, and it passed — because the model carried a ninety-degree phase
// error that moved the peak onto that time. It does not belong there. The true
// response has a sharp onset followed by a long tail, so the largest excursion
// lags the travel time by an offset set by the source duration and the tail,
// not by range: about 46 ms in this medium, which at 2 m is four times the
// travel time itself and at 30 m is a quarter of it. An apparent velocity from
// peak picking is therefore badly range dependent, which is a fact WP3 needs
// and the reverse of what the old test asserted.
//
// A differential lag has none of that. The source, the tail shape and the
// instrument all cancel between two ranges, leaving the moveout — which is what
// an array measures anyway, and which internal/fk and this model agree on to
// within a percent.
func TestMoveoutMatchesRayleighVelocity(t *testing.T) {
	const fs = 2000.0
	g := HalfSpaceGF{Soil: soil.Loam()}
	const lead = 400

	force := make([]units.Newtons, int(2*fs))
	for i := range 100 {
		force[lead+i] = units.Newtons(1000 * math.Sin(math.Pi*float64(i)/100))
	}
	trace := func(r units.Metres) []float64 {
		vel, err := g.Synthesise(force, fs, r)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, len(vel))
		for i, v := range vel {
			out[i] = float64(v)
		}
		return out
	}

	cr, err := g.Soil.RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}
	ranges := []units.Metres{10, 20, 30}
	traces := make([][]float64, len(ranges))
	for i, r := range ranges {
		traces[i] = trace(r)
	}
	for i := 1; i < len(ranges); i++ {
		d := float64(ranges[i] - ranges[i-1])
		lag := float64(crossLag(traces[i-1], traces[i])) / fs
		apparent := d / lag
		t.Logf("%g -> %g m: lag %.4f s, moveout velocity %.1f m/s against cR %.1f m/s",
			ranges[i-1], ranges[i], lag, apparent, cr)
		if rel := math.Abs(apparent-float64(cr)) / float64(cr); rel > 0.03 {
			t.Errorf("%g -> %g m: moveout velocity %.1f m/s, want %.1f within 3%%",
				ranges[i-1], ranges[i], apparent, cr)
		}
	}

	// And the fact the old test had backwards, recorded rather than asserted
	// away: the peak lags the travel time, by an amount that is nearly
	// independent of range.
	for _, r := range []units.Metres{5, 10, 20, 30} {
		x := trace(r)
		peak, at := 0.0, 0
		for i, v := range x {
			if math.Abs(v) > peak {
				peak, at = math.Abs(v), i
			}
		}
		travel, err := g.TravelTime(r)
		if err != nil {
			t.Fatal(err)
		}
		got := float64(at-lead) / fs
		t.Logf("r=%4.0f m: peak at %.4f s against a travel time of %.4f s — %.1f ms late, "+
			"an apparent velocity of %.0f m/s", r, got, float64(travel), 1e3*(got-float64(travel)), float64(r)/got)
	}
}

// crossLag is the sample shift at which b best matches a.
func crossLag(a, b []float64) int {
	best, at := math.Inf(-1), 0
	for k := range len(a) / 2 {
		var acc float64
		for i := k; i < len(a); i++ {
			acc += b[i] * a[i-k]
		}
		if acc > best {
			best, at = acc, k
		}
	}
	return at
}
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

// A horizontal force excites the Rayleigh wave through the horizontal
// eigenfunction at the source and the vertical one at the receiver, where a
// vertical force uses the vertical one twice. So the amplitude ratio between
// them is the surface ellipticity, and the phase differs by exactly a quarter
// cycle. Both are consequences of the mode shapes, not choices.
func TestRadialForceResponseIsEllipticityTimesQuarterCycle(t *testing.T) {
	for name, h := range map[string]soil.HalfSpace{
		"Poisson solid": soil.PoissonSolid(200, 1900),
		"dry sand":      soil.DrySand(),
		"loam":          soil.Loam(),
	} {
		t.Run(name, func(t *testing.T) {
			g := HalfSpaceGF{Soil: h}
			for _, f := range []units.Hertz{5, 20, 100} {
				vert, err := g.VelocityResponse(10, f)
				if err != nil {
					t.Fatal(err)
				}
				rad, err := g.RadialForceResponse(10, f)
				if err != nil {
					t.Fatal(err)
				}
				e, err := NewEigenfunctions(h, 2*math.Pi*float64(f))
				if err != nil {
					t.Fatal(err)
				}
				if got, want := cmplx.Abs(rad)/cmplx.Abs(vert), e.Ellipticity(); math.Abs(got-want) > 1e-12 {
					t.Errorf("%g Hz: amplitude ratio %g, want the ellipticity %g", f, got, want)
				}
				dphase := (cmplx.Phase(rad) - cmplx.Phase(vert)) * 180 / math.Pi
				for dphase > 180 {
					dphase -= 360
				}
				for dphase < -180 {
					dphase += 360
				}
				if math.Abs(math.Abs(dphase)-90) > 1e-9 {
					t.Errorf("%g Hz: phase difference %g degrees, want a quarter cycle", f, dphase)
				}
			}
		})
	}
	// For a Poisson solid the ratio is the textbook 0.68127.
	g := HalfSpaceGF{Soil: soil.PoissonSolid(200, 1900)}
	v, _ := g.VelocityResponse(10, 20)
	r, _ := g.RadialForceResponse(10, 20)
	if got := cmplx.Abs(r) / cmplx.Abs(v); math.Abs(got-0.68127) > 1e-4 {
		t.Errorf("Poisson solid ratio %.6f, want 0.68127", got)
	}
}

// Only the radial component of a horizontal force drives the Rayleigh wave's
// vertical motion. The transverse component excites Love waves, which have no
// vertical component at all, so a sensor directly abeam of a walker hears
// nothing from their fore-aft shear.
func TestHorizontalForceFollowsCosineAzimuth(t *testing.T) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	const (
		r = 10.0
		f = 20.0
	)
	// A horizontal force along +x, receiver swept around the source.
	force := [3]float64{1, 0, 0}
	inline, err := g.ResponseToForce(r, 0, force, f)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name       string
		dx, dy     float64
		wantFactor float64
	}{
		{"in line, ahead", r, 0, 1},
		{"in line, behind", -r, 0, -1},
		{"abeam", 0, r, 0},
		{"forty-five degrees", r / math.Sqrt2, r / math.Sqrt2, 1 / math.Sqrt2},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := g.ResponseToForce(c.dx, c.dy, force, f)
			if err != nil {
				t.Fatal(err)
			}
			want := complex(c.wantFactor, 0) * inline
			if d := cmplx.Abs(got - want); d > 1e-12*cmplx.Abs(inline)+1e-30 {
				t.Errorf("response %v, want %v (cos azimuth = %g)", got, want, c.wantFactor)
			}
		})
	}
}

// A pure vertical force must reproduce VelocityResponse exactly, whatever the
// azimuth, since a vertical force has no direction in the horizontal plane.
func TestResponseToForceMatchesVerticalResponse(t *testing.T) {
	g := HalfSpaceGF{Soil: soil.Loam()}
	want, err := g.VelocityResponse(10, 25)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range [][2]float64{{10, 0}, {0, 10}, {-6, 8}, {8, -6}} {
		got, err := g.ResponseToForce(p[0], p[1], [3]float64{0, 0, 1}, 25)
		if err != nil {
			t.Fatal(err)
		}
		if cmplx.Abs(got-want) > 1e-18 {
			t.Errorf("offset %v: %v, want %v", p, got, want)
		}
	}
	if _, err := g.ResponseToForce(0, 0, [3]float64{0, 0, 1}, 25); err == nil {
		t.Error("expected an error when the receiver coincides with the source")
	}
}
