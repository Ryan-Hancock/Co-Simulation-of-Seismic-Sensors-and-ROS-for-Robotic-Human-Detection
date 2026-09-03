package fdtd

import (
	"math"
	"math/cmplx"
	"testing"

	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/fk"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
	"geosim.dev/geosim/internal/visco"
)

// V5 is the load-bearing validation of this slice: the layered
// frequency-wavenumber solve is right because a method sharing none of its code
// gets the same answer.
//
// The two paths agree on nothing except the physics. One expands the field in
// wavenumber, solves a boundary-value problem with propagator matrices at each
// one and integrates a Hankel transform; the other steps a difference stencil
// forward in time and never forms a wavenumber at all. There is no shared
// constant, no shared root finder, no shared quadrature. A sign error in a
// layer propagator cannot survive this comparison, and neither can a factor
// hidden in the source normalisation, because the amplitudes are absolute on
// both sides.
//
// The medium is a comparison medium, not a soil — see internal/visco. Both
// paths implement one standard linear solid exactly, so no part of the residual
// is a difference in what is being modelled.

const (
	v5Freq    = 25.0 // wavelet peak frequency
	v5Band    = 90.0 // where the wavelet's spectrum is negligible
	v5RefFreq = 30.0
	v5Q       = 25.0
	// The reference needs many more wavenumbers than the default. The
	// integrand has a near-pole at the Rayleigh wavenumber, where a
	// trapezoidal rule is only first-order accurate, and the default sampling
	// leaves about two percent at the top of the band — which is the size of
	// the disagreement being measured.
	v5RefSamples = 160000
)

// synthesise drives the frequency-wavenumber path with exactly the wavelet the
// grid was driven with, sampled on exactly the instants the grid recorded.
func synthesise(t *testing.T, st layer.Stack, sls visco.SLS, ranges []units.Metres,
	peak units.Newtons, dt, t0 float64, n int) [][]float64 {
	t.Helper()
	m := fk.Medium{Stack: st, Relax: &sls, RefFreq: v5RefFreq}
	nfft := dsp.NextPow2(n)
	fs := 1 / dt
	df := fs / float64(nfft)
	coeff := make([][]complex128, len(ranges))
	for i := range coeff {
		coeff[i] = make([]complex128, nfft/2+1)
	}
	for b := 1; b <= nfft/2; b++ {
		f := float64(b) * df
		if f > v5Band {
			break
		}
		u, err := m.VerticalDisplacementMulti(ranges, f, fk.Integration{Samples: v5RefSamples})
		if err != nil {
			t.Fatal(err)
		}
		w := RickerSpectrum(peak, v5Freq, f)
		// Displacement to velocity, then the half-step offset between the
		// stress grid and the velocity grid. Forgetting the second is a pure
		// phase ramp, which looks exactly like a wrong wave speed.
		jw := complex(0, 2*math.Pi*f)
		shift := cmplx.Exp(complex(0, -2*math.Pi*f*t0))
		for i := range ranges {
			coeff[i][b] = u[i] * w * jw * shift * complex(fs, 0)
		}
	}
	out := make([][]float64, len(ranges))
	for i := range ranges {
		out[i] = dsp.IRFFT(coeff[i], nfft)[:n]
	}
	return out
}

// agreement is how well one receiver's two traces match.
type agreement struct {
	rng      units.Metres
	peak     float64 // ratio of peak amplitudes, grid over wavenumber
	lag      float64 // seconds the grid trace runs late
	residual float64 // rms of what is left once the lag is removed
}

func compare(got []units.Velocity, want []float64, dt float64, rng units.Metres) agreement {
	a := agreement{rng: rng}
	var pa, pb float64
	for i := range want {
		pa = math.Max(pa, math.Abs(float64(got[i])))
		pb = math.Max(pb, math.Abs(want[i]))
	}
	a.peak = pa / pb

	// The lag is found by cross-correlation rather than by comparing arrival
	// times, so it measures the whole waveform and not one picked sample.
	maxLag := len(want) / 8
	best, bestAt := math.Inf(-1), 0
	corr := func(k int) float64 {
		var s float64
		for i := range want {
			j := i - k
			if j >= 0 && j < len(want) {
				s += float64(got[i]) * want[j]
			}
		}
		return s
	}
	for k := -maxLag; k <= maxLag; k++ {
		if c := corr(k); c > best {
			best, bestAt = c, k
		}
	}
	if l, r := corr(bestAt-1), corr(bestAt+1); l-2*best+r != 0 {
		a.lag = (float64(bestAt) - 0.5*(r-l)/(l-2*best+r)) * dt
	} else {
		a.lag = float64(bestAt) * dt
	}

	var num, den float64
	for i := range want {
		j := i - bestAt
		v := 0.0
		if j >= 0 && j < len(want) {
			v = want[j]
		}
		d := float64(got[i]) - v
		num += d * d
		den += v * v
	}
	a.residual = math.Sqrt(num / den)
	return a
}

// shoot runs one grid and compares it with the wavenumber path.
func shoot(t *testing.T, st layer.Stack, sls visco.SLS, spacing units.Metres,
	rMax, depth units.Metres, ranges []units.Metres, seconds float64) []agreement {
	t.Helper()
	m := Model{
		Stack: st, Relax: sls, RefFreq: v5RefFreq,
		MaxRange: rMax, Depth: depth, Spacing: spacing, DominantFreq: v5Freq,
	}
	s, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	dt := s.Dt()
	steps := int(seconds / dt)
	force, err := Ricker(1000, v5Freq, dt, steps)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Run(Shot{Model: m, Force: force, Ranges: ranges, Steps: steps})
	if err != nil {
		t.Fatal(err)
	}
	snapped := make([]units.Metres, len(res.Traces))
	for i, tr := range res.Traces {
		snapped[i] = tr.Range
	}
	ref := synthesise(t, st, sls, snapped, 1000, dt, res.T0, steps)
	out := make([]agreement, len(res.Traces))
	for i, tr := range res.Traces {
		out[i] = compare(tr.Vertical, ref[i], dt, tr.Range)
	}
	return out
}

// richardson extrapolates two first-order-accurate results to zero spacing.
//
// First order, not second, because that is what the scheme measurably is here:
// the free surface is imposed on a single row, which makes the surface a
// first-order feature of an otherwise second-order stencil, and a Rayleigh wave
// lives entirely on that surface. TestConvergenceIsFirstOrder pins the exponent
// rather than assuming it, and extrapolating with the wrong exponent would
// produce a confident wrong answer.
func richardson(coarse, fine float64) float64 { return 2*fine - coarse }

func report(t *testing.T, label string, coarse, fine []agreement, tol float64) {
	t.Helper()
	for i := range fine {
		p := richardson(coarse[i].peak, fine[i].peak)
		l := richardson(coarse[i].lag, fine[i].lag)
		r := richardson(coarse[i].residual, fine[i].residual)
		t.Logf("%s r=%5.2f m: peak ratio %.4f -> %.4f (extrapolated %.4f), "+
			"lag %+.3f -> %+.3f ms (extrapolated %+.3f ms), residual %.2f%% -> %.2f%% (extrapolated %.2f%%)",
			label, fine[i].rng, coarse[i].peak, fine[i].peak, p,
			1e3*coarse[i].lag, 1e3*fine[i].lag, 1e3*l, 100*coarse[i].residual, 100*fine[i].residual, 100*r)
		if math.Abs(p-1) > tol {
			t.Errorf("%s r=%g m: extrapolated peak ratio %.4f, want 1 within %.1f%%",
				label, fine[i].rng, p, 100*tol)
		}
		if math.Abs(r) > tol {
			t.Errorf("%s r=%g m: extrapolated residual %.2f%%, want below %.1f%%",
				label, fine[i].rng, 100*r, 100*tol)
		}
	}
}

// V5 for a homogeneous half-space. The layered solver reduces to the case the
// analytic model already covers, so a failure here is in the grid or in the
// quadrature rather than in the layer handling.
func TestV5Homogeneous(t *testing.T) {
	if testing.Short() {
		t.Skip("the grid runs are minutes, not seconds")
	}
	sls, err := visco.Fit(v5Q, v5RefFreq)
	if err != nil {
		t.Fatal(err)
	}
	st := layer.Uniform(soil.Loam())
	ranges := []units.Metres{4, 8, 16}
	coarse := shoot(t, st, sls, 0.111, 26, 18, ranges, 0.3)
	fine := shoot(t, st, sls, 0.0555, 26, 18, ranges, 0.3)
	report(t, "homogeneous", coarse, fine, 0.02)
}

// V5 for a layered medium: a soft surface layer over a stiffer half-space,
// which is the case the whole of slice 3 exists for.
//
// The contrast is deliberately strong. A gentle one would produce a response
// close enough to the homogeneous answer that agreement would say little about
// the layer machinery; at this contrast the surface wave is strongly dispersive
// and the analytic half-space model is nowhere near it.
func TestV5Layered(t *testing.T) {
	if testing.Short() {
		t.Skip("the grid runs are minutes, not seconds")
	}
	sls, err := visco.Fit(v5Q, v5RefFreq)
	if err != nil {
		t.Fatal(err)
	}
	st := layer.Stack{
		{Thickness: 2, Vp: 350, Vs: 140, Density: 1600},
		{Vp: 800, Vs: 320, Density: 2000},
	}
	ranges := []units.Metres{3, 6, 10}
	coarse := shoot(t, st, sls, 0.1, 16, 10, ranges, 0.3)
	fine := shoot(t, st, sls, 0.05, 16, 10, ranges, 0.3)
	report(t, "layered", coarse, fine, 0.03)
}

// The extrapolation above is only as good as the exponent it assumes, so the
// exponent is measured.
//
// Three spacings, each a halving of the last. A second-order scheme would show
// the error falling by four; this one shows it falling by two, and knowing
// which is the difference between an extrapolation that lands on the right
// answer and one that overshoots it by the same amount it corrects.
func TestConvergenceIsFirstOrder(t *testing.T) {
	if testing.Short() {
		t.Skip("three grid runs")
	}
	sls, err := visco.Fit(v5Q, v5RefFreq)
	if err != nil {
		t.Fatal(err)
	}
	st := layer.Uniform(soil.Loam())
	ranges := []units.Metres{8}
	var errs []float64
	for _, h := range []units.Metres{0.222, 0.111, 0.0555} {
		a := shoot(t, st, sls, h, 20, 14, ranges, 0.3)
		errs = append(errs, math.Abs(a[0].peak-1))
		t.Logf("h=%.4f m: peak ratio %.4f, residual %.2f%%", h, a[0].peak, 100*a[0].residual)
	}
	for i := 1; i < len(errs); i++ {
		order := math.Log2(errs[i-1] / errs[i])
		t.Logf("halving the spacing reduced the amplitude error by %.2fx (order %.2f)",
			errs[i-1]/errs[i], order)
		if order < 0.7 || order > 1.6 {
			t.Errorf("observed convergence order %.2f, expected first order — "+
				"richardson() extrapolates on the assumption that it is", order)
		}
	}
}
