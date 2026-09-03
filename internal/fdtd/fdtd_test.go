package fdtd

import (
	"math"
	"testing"

	"math/cmplx"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
	"geosim.dev/geosim/internal/visco"
)

// peakAt returns the largest absolute sample and the time of it, refined by
// fitting a parabola through the sample and its neighbours.
//
// Without the refinement an arrival time is quantised to the time step, which
// for the grids here is a percent of the travel time — the same size as the
// error being measured.
func peakAt(tr Trace, dt, t0 float64) (amp, at float64) {
	best := 0
	for i, v := range tr.Vertical {
		if math.Abs(float64(v)) > math.Abs(float64(tr.Vertical[best])) {
			best = i
		}
	}
	amp = math.Abs(float64(tr.Vertical[best]))
	at = t0 + float64(best)*dt
	if best > 0 && best < len(tr.Vertical)-1 {
		a := math.Abs(float64(tr.Vertical[best-1]))
		c := math.Abs(float64(tr.Vertical[best+1]))
		if d := a - 2*amp + c; d != 0 {
			at += -0.5 * (c - a) / d * dt
		}
	}
	return amp, at
}

// The load-bearing absolute check, and the one that shares nothing at all with
// the frequency-domain path: hold a steady vertical force on the surface and
// the static displacement it settles to is Boussinesq's.
//
// This pins the source normalisation — a point force spread over the axis cell
// has an area in it, and getting that area wrong scales every amplitude the
// package will ever produce by a constant that no relative comparison would
// reveal. It also exercises the axis condition, the free surface and the
// half-cell extrapolation at once, because all three sit between the applied
// traction and the recorded number.
//
// The domain has to be large. A perfectly matched layer absorbs waves; a static
// field does not propagate, so the layer does not absorb it, it eats it, and
// the displacement drifts downward at a rate set by how much of the static
// field lies inside the boundary. At four times the furthest receiver the drift
// is under a percent over the settling window; at eleven times it is
// unmeasurable.
func TestBoussinesqStaticLimit(t *testing.T) {
	h := soil.Loam()
	const rMax = 40.0
	m := Model{Stack: layer.Uniform(h), MaxRange: rMax, Depth: rMax, DominantFreq: 25}
	sp, err := m.SpacingFor(75, 15)
	if err != nil {
		t.Fatal(err)
	}
	m.Spacing = sp
	s, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	dt := s.Dt()
	steps := int(0.3 / dt)

	// A step, smoothed enough for the grid to carry it.
	const load = 1000.0
	sigma := 1 / (2 * math.Pi * 25)
	force := make([]float64, steps)
	for i := range force {
		force[i] = load * 0.5 * (1 + math.Erf((float64(i)*dt-5*sigma)/(math.Sqrt2*sigma)))
	}
	res, err := Run(Shot{Model: m, Force: force, Ranges: []units.Metres{1, 2, 3, 5}, Steps: steps})
	if err != nil {
		t.Fatal(err)
	}

	mu, nu := float64(h.ShearModulus()), h.PoissonRatio()
	for _, tr := range res.Traces {
		u, settled := 0.0, 0.0
		for i, v := range tr.Vertical {
			u += float64(v) * dt
			if i == steps-1 {
				settled = u
			}
		}
		want := load * (1 - nu) / (2 * math.Pi * mu * float64(tr.Range))
		ratio := settled / want
		t.Logf("r=%4.1f m: static displacement %.4e m, Boussinesq %.4e m, ratio %.4f",
			tr.Range, settled, want, ratio)
		if math.Abs(ratio-1) > 0.01 {
			t.Errorf("r=%g m: static displacement is %.4f of Boussinesq's, want within 1%%", tr.Range, ratio)
		}
	}
}

// The Courant factor has to be a real limit, not a comfortable margin nobody
// has ever tested. A scheme that is stable at every factor anyone tries is one
// whose time step is being set by something other than the stability condition.
func TestStabilityLimit(t *testing.T) {
	run := func(courant float64) float64 {
		m := Model{
			Stack: layer.Uniform(soil.Loam()), MaxRange: 6, Depth: 6,
			Spacing: 0.15, DominantFreq: 30, Courant: courant,
		}
		s, err := New(m)
		if err != nil {
			t.Fatal(err)
		}
		steps := int(0.15 / s.Dt())
		force, _ := Ricker(1000, 30, s.Dt(), steps)
		s.Drive(force)
		for range steps {
			s.Step()
		}
		return s.MaxSpeed()
	}
	stable := run(defaultCourant)
	if math.IsNaN(stable) || stable > 1 {
		t.Fatalf("the default Courant factor is not stable: peak speed %g m/s", stable)
	}
	// The plain two-dimensional limit is a factor of one; the axis terms pull
	// it below that, so a factor comfortably above one must diverge.
	if bad := run(1.4); !math.IsNaN(bad) && bad < 1e3*stable {
		t.Errorf("Courant 1.4 did not diverge (peak speed %g m/s against %g m/s stable) — "+
			"the stability limit is not where timeStep says it is", bad, stable)
	}
}

// The surface wave must travel at the Rayleigh velocity of the medium, which
// internal/soil computes from the secular equation and this package has never
// heard of.
//
// Measured as a slope over several ranges rather than as a single travel time.
// A single time also contains the source delay and the offset between the
// wavelet's onset and its peak; a slope contains neither, so it tests the
// propagation and nothing else.
func TestRayleighVelocity(t *testing.T) {
	h := soil.Loam()
	m := Model{Stack: layer.Uniform(h), MaxRange: 22, Depth: 16, DominantFreq: 30}
	sp, _ := m.SpacingFor(90, 20)
	m.Spacing = sp
	s, _ := New(m)
	steps := int(0.35 / s.Dt())
	force, _ := Ricker(1000, 30, s.Dt(), steps)
	ranges := []units.Metres{8, 10, 12, 14, 16}
	res, err := Run(Shot{Model: m, Force: force, Ranges: ranges, Steps: steps})
	if err != nil {
		t.Fatal(err)
	}

	var sx, sy, sxx, sxy float64
	n := float64(len(res.Traces))
	for _, tr := range res.Traces {
		_, at := peakAt(tr, res.Dt, res.T0)
		x, y := float64(tr.Range), at
		sx, sy, sxx, sxy = sx+x, sy+y, sxx+x*x, sxy+x*y
	}
	slope := (n*sxy - sx*sy) / (n*sxx - sx*sx)
	got := 1 / slope
	want, err := h.RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Rayleigh velocity: grid %.2f m/s, secular equation %.2f m/s (%.2f%%)",
		got, want, 100*(got/float64(want)-1))
	if math.Abs(got/float64(want)-1) > 0.02 {
		t.Errorf("surface wave travels at %.2f m/s, want %.2f m/s within 2%%", got, want)
	}
}

// How much energy the absorbing layer sends back.
//
// Measured against the same shot in a domain three times as wide, where the
// boundary is too far away to return anything within the window. The difference
// between the two traces is the reflection and nothing else, since every other
// aspect of the two runs is identical — same spacing, same step, same source.
//
// The Rayleigh wave is the hard case and the reason the layer is a C-PML rather
// than a classical one: it reaches the outer boundary travelling along it, at
// grazing incidence, where an unshifted profile reflects a large fraction of
// what it receives.
func TestPMLReflection(t *testing.T) {
	shoot := func(rMax units.Metres) *Result {
		m := Model{Stack: layer.Uniform(soil.Loam()), MaxRange: rMax, Depth: 14, DominantFreq: 30}
		m.Spacing = 0.15
		s, _ := New(m)
		steps := int(0.45 / s.Dt())
		force, _ := Ricker(1000, 30, s.Dt(), steps)
		res, err := Run(Shot{Model: m, Force: force, Ranges: []units.Metres{4, 8}, Steps: steps})
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	near, far := shoot(12), shoot(36)
	for i := range near.Traces {
		var peak, diff float64
		for n := range near.Traces[i].Vertical {
			a := float64(near.Traces[i].Vertical[n])
			b := float64(far.Traces[i].Vertical[n])
			peak = math.Max(peak, math.Abs(b))
			diff = math.Max(diff, math.Abs(a-b))
		}
		db := 20 * math.Log10(diff/peak)
		t.Logf("r=%4.1f m: boundary returns %.2e of a peak of %.2e (%.1f dB)",
			near.Traces[i].Range, diff, peak, db)
		if db > -40 {
			t.Errorf("r=%g m: absorbing layer reflects at %.1f dB, want below -40 dB",
				near.Traces[i].Range, db)
		}
	}
}

// dft is one frequency of a trace, evaluated directly. Only a handful of bins
// are ever wanted, and a direct sum avoids having to reason about how a
// power-of-two pad interacts with an arrival that has not finished.
func dft(v []units.Velocity, f, dt, t0 float64) complex128 {
	var acc complex128
	for i, x := range v {
		phi := -2 * math.Pi * f * (t0 + float64(i)*dt)
		acc += complex(float64(x), 0) * complex(math.Cos(phi), math.Sin(phi))
	}
	return acc * complex(dt, 0)
}

// Attenuation must attenuate by the amount the material model says, and the
// scheme's memory variables are the only place that could go wrong.
//
// The comparison is against the same shot in the same medium with the
// relaxation switched off, taken as a ratio between two ranges. Geometric
// spreading is much the larger effect and cancels in the first ratio;
// everything that depends on the medium's stiffness rather than on distance —
// the excitation, the modulus in the denominator — cancels in the second. What
// survives is the dissipation over the range difference and nothing else.
func TestAttenuationMatchesTheMaterialModel(t *testing.T) {
	const f0, qTarget, fRef = 30.0, 25.0, 30.0
	sls, err := visco.Fit(qTarget, fRef)
	if err != nil {
		t.Fatal(err)
	}
	ranges := []units.Metres{6, 18}
	shoot := func(relax visco.SLS) *Result {
		m := Model{
			Stack: layer.Uniform(soil.Loam()), Relax: relax, RefFreq: fRef,
			MaxRange: 26, Depth: 18, DominantFreq: f0, Spacing: 0.12,
		}
		s, err := New(m)
		if err != nil {
			t.Fatal(err)
		}
		steps := int(0.45 / s.Dt())
		force, _ := Ricker(1000, f0, s.Dt(), steps)
		res, err := Run(Shot{Model: m, Force: force, Ranges: ranges, Steps: steps})
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	elastic, lossy := shoot(visco.SLS{}), shoot(sls)

	cR, err := soil.Loam().RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}
	dr := float64(ranges[1] - ranges[0])
	for _, f := range []float64{15, 30, 50} {
		ratio := func(res *Result) float64 {
			a := cmplx.Abs(dft(res.Traces[0].Vertical, f, res.Dt, res.T0))
			b := cmplx.Abs(dft(res.Traces[1].Vertical, f, res.Dt, res.T0))
			return b / a
		}
		got := ratio(lossy) / ratio(elastic)
		// The Rayleigh velocity is homogeneous of degree one in the layer
		// velocities, so a scalar relaxation carries it along unchanged: the
		// complex Rayleigh velocity is the nominal one put through the same
		// relaxation.
		k := complex(2*math.Pi*f, 0) / sls.Velocity(float64(cR), fRef, f)
		want := math.Exp(imag(k) * dr)
		t.Logf("%4.0f Hz: measured decay over %g m %.4f, material model %.4f (Q = %.1f)",
			f, dr, got, want, sls.Q(f))
		if math.Abs(got/want-1) > 0.05 {
			t.Errorf("%g Hz: decay %.4f, want %.4f within 5%%", f, got, want)
		}
	}
}

// Elastic runs must be exactly elastic, not nearly so. The zero value of a
// visco.SLS has both relaxation times zero, and a scheme that treated that as
// a very short relaxation instead of as no relaxation would dissipate.
func TestZeroRelaxationIsElastic(t *testing.T) {
	m := Model{Stack: layer.Uniform(soil.Loam()), MaxRange: 6, Depth: 6, Spacing: 0.15, DominantFreq: 30}
	s, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	if s.lossy {
		t.Error("the zero relaxation was treated as dissipative")
	}
	if s.rrr != nil {
		t.Error("memory variables were allocated for an elastic run")
	}
	if _, err := New(Model{
		Stack: layer.Uniform(soil.Loam()), MaxRange: 6, Depth: 6, Spacing: 0.15,
		Relax: visco.SLS{TauSigma: 1e-2, TauEps: 1e-3},
	}); err == nil {
		t.Error("expected an error for a solid that would amplify rather than dissipate")
	}
}
