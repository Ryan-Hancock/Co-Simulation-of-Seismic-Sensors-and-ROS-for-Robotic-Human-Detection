package fk

import (
	"math"
	"math/cmplx"
	"testing"

	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

func lossless(h soil.HalfSpace) Medium {
	st := layer.Uniform(h)
	st[0].Qs = 1e9
	return Medium{Stack: st, DefaultQ: 1e9}
}

// The absolute normalisation, pinned against a closed form.
//
// As the wavenumber grows the response tends to C/k, and C is the coefficient
// of the medium's static near-field behaviour: for a half-space, exactly
// Boussinesq's (1-nu)/(2*pi*mu). That single number fixes the source
// normalisation, the Hankel convention, the scaling of the motion-stress
// vector and the traction boundary condition all at once — every factor of
// 2*pi that could have been dropped anywhere shows up here.
//
// This is what slice 0 could not do. Its amplitude came from a surface-wave
// excitation formula whose prefactor had no independent check, so the model was
// correct in its range, frequency and attenuation dependence but unpinned in
// absolute scale. It is pinned now.
func TestLargeWavenumberLimitIsBoussinesq(t *testing.T) {
	for name, h := range map[string]soil.HalfSpace{
		"loam":           soil.Loam(),
		"dry sand":       soil.DrySand(),
		"weathered rock": soil.WeatheredRock(),
	} {
		t.Run(name, func(t *testing.T) {
			m := lossless(h)
			want := (1 - h.PoissonRatio()) / (2 * math.Pi * float64(h.ShearModulus()))
			for _, k := range []float64{1, 10, 200} {
				u, err := m.SurfaceResponse(k, 0.05)
				if err != nil {
					t.Fatal(err)
				}
				got := k * cmplx.Abs(u)
				// A part in a thousand rather than machine precision. Far into
				// the evanescent regime the P and SV vertical wavenumbers both
				// approach k, so the two eigenvectors nearly coincide and the
				// basis the propagator is built in becomes ill-conditioned.
				// The effect is worst in the stiffest media, where a given
				// wavenumber is further from the body-wave branches; weathered
				// rock at k=200 loses about four digits. It is a real
				// limitation of an eigenvector formulation, and the reason the
				// asymptote is sampled at a few hundred inverse metres rather
				// than at the largest wavenumber available.
				if rel := math.Abs(got-want) / want; rel > 1e-3 {
					t.Errorf("k=%g: k*|u| = %.6e, Boussinesq coefficient is %.6e (rel %.2e)", k, got, want, rel)
				}
			}
		})
	}
}

// And the integral of it: at a frequency low enough to be effectively static,
// the displacement at range must be Boussinesq's.
//
// This exercises the whole path — response, asymptote subtraction, quadrature —
// against a number that owes nothing to any of it.
func TestStaticLimitIsBoussinesq(t *testing.T) {
	h := soil.Loam()
	m := lossless(h)
	mu := float64(h.ShearModulus())
	nu := h.PoissonRatio()

	for _, r := range []units.Metres{2, 5, 10, 20} {
		want := (1 - nu) / (2 * math.Pi * mu * float64(r))
		got, err := m.VerticalDisplacement(r, 0.02, Integration{Samples: 8000})
		if err != nil {
			t.Fatal(err)
		}
		if rel := math.Abs(cmplx.Abs(got)-want) / want; rel > 0.02 {
			t.Errorf("r=%g m: %.6e m/N, Boussinesq gives %.6e (rel %.3f)", r, cmplx.Abs(got), want, rel)
		}
	}
}

// The point of the slice, measured.
//
// Slice 0 kept only the far-field Rayleigh pole, and the plan predicted that
// would fail inside about a wavelength — the regime a robot detecting someone
// at a few metres actually occupies. It does: within a tenth of a wavelength
// the far-field model is low by a factor of three, and it is still 20% low at
// half a wavelength. Beyond about one wavelength the two agree.
//
// The direction matters as much as the size. The far-field model
// *underestimates*, because the near-field terms it omits add to the signal
// rather than cancelling it. A detector tuned on far-field predictions would
// be conservative at close range, not optimistic — but the localisation and
// amplitude inversion in WP3 would be biased.
func TestFarFieldModelFailsInsideOneWavelength(t *testing.T) {
	h := soil.Loam()
	m := Medium{Stack: layer.Uniform(h)}
	g := green.HalfSpaceGF{Soil: h}
	cr, err := h.RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}

	ratio := func(r units.Metres, f float64) float64 {
		u, err := m.VerticalDisplacement(r, f, Integration{Samples: 6000})
		if err != nil {
			t.Fatal(err)
		}
		full := cmplx.Abs(complex(0, 2*math.Pi*f) * u)
		far, err := g.VelocityResponse(r, units.Hertz(f))
		if err != nil {
			t.Fatal(err)
		}
		return full / cmplx.Abs(far)
	}

	const f = 10.0
	lambda := float64(cr) / f

	t.Run("far field is badly low well inside a wavelength", func(t *testing.T) {
		got := ratio(units.Metres(0.05*lambda), f)
		if got < 2 {
			t.Errorf("at 0.05 wavelengths the full solution is only %.2f times the far-field model; expected at least 2", got)
		}
	})

	t.Run("still low at half a wavelength", func(t *testing.T) {
		got := ratio(units.Metres(0.3*lambda), f)
		if got < 1.15 {
			t.Errorf("at 0.3 wavelengths the ratio is %.3f; the near field should still be adding appreciably", got)
		}
	})

	t.Run("they agree beyond a wavelength", func(t *testing.T) {
		// Not exactly: body waves interfere with the Rayleigh arrival and
		// modulate the amplitude by some tens of percent with range, which a
		// Rayleigh-only model cannot produce at all. The bound is on the bias,
		// not on the scatter.
		var sum float64
		var n int
		for _, mult := range []float64{1.2, 1.6, 2.1, 2.7} {
			sum += ratio(units.Metres(mult*lambda), f)
			n++
		}
		if mean := sum / float64(n); mean < 0.85 || mean > 1.15 {
			t.Errorf("beyond a wavelength the mean ratio is %.3f; the two models should agree on average", mean)
		}
	})

	t.Run("the departure grows monotonically as range shrinks", func(t *testing.T) {
		var prev float64
		for _, mult := range []float64{0.6, 0.4, 0.25, 0.15, 0.08} {
			got := ratio(units.Metres(mult*lambda), f)
			if prev > 0 && got <= prev {
				t.Errorf("at %.2f wavelengths the ratio is %.3f, not above the %.3f at the previous, larger range", mult, got, prev)
			}
			prev = got
		}
	})
}

// A layered medium's near field is governed by its surface layer, because short
// wavelengths never reach the deeper material. So the static coefficient of a
// layered stack must match that of a half-space made of its top layer alone —
// a property of the physics that also confirms the layer propagators are not
// leaking the half-space's constants upward.
func TestStaticCoefficientFollowsTheSurfaceLayer(t *testing.T) {
	top := layer.Layer{Thickness: 3, Vp: 400, Vs: 160, Density: 1600, Qs: 1e9}
	stacked := Medium{
		Stack:    layer.Stack{top, {Vp: 2200, Vs: 1100, Density: 2400, Qs: 1e9}},
		DefaultQ: 1e9,
	}
	alone := Medium{Stack: layer.Stack{{Vp: top.Vp, Vs: top.Vs, Density: top.Density, Qs: 1e9}}, DefaultQ: 1e9}

	// Far enough into the near field that the wave cannot feel three metres
	// down: a wavelength well under the layer thickness.
	const k = 40.0
	a, err := stacked.StaticCoefficient(1, k)
	if err != nil {
		t.Fatal(err)
	}
	b, err := alone.StaticCoefficient(1, k)
	if err != nil {
		t.Fatal(err)
	}
	if rel := cmplx.Abs(a-b) / cmplx.Abs(b); rel > 1e-6 {
		t.Errorf("layered stack gives %.6e, its surface layer alone gives %.6e (rel %.2e); the near field should not see the half-space",
			cmplx.Abs(a), cmplx.Abs(b), rel)
	}
}

// The quadrature has to be converged enough that the physics above is not
// measuring the integration, and the way to know that is the observed order —
// not a tolerance, which can be met by an accident of where the error happens
// to sit at one sample count.
//
// This test replaced one that asserted a two percent agreement and passed while
// the rule was carrying a two percent error. Slice 4 found it by comparing
// against a completely separate solver, which is a long way to travel for
// something an order check catches for free.
//
// The rule is second order, so halving the spacing must quarter the error. The
// failure this guards against is losing the closed panel at k = 0, where the
// integrand is -C rather than zero: dropping half of that panel is an error
// linear in the spacing, so it does not merely make the answer worse, it
// changes the convergence order and swamps everything else.
func TestQuadratureIsSecondOrder(t *testing.T) {
	m := Medium{Stack: layer.Uniform(soil.Loam())}
	const r, f = units.Metres(16), 60.0
	g, err := m.GridFor([]units.Metres{r}, f, Integration{})
	if err != nil {
		t.Fatal(err)
	}
	// Richardson on the two finest, which for a second-order rule removes the
	// leading term and leaves a limit far more accurate than either.
	fine, err := m.VerticalDisplacement(r, f, Integration{Samples: 16 * g.Samples})
	if err != nil {
		t.Fatal(err)
	}
	finer, err := m.VerticalDisplacement(r, f, Integration{Samples: 32 * g.Samples})
	if err != nil {
		t.Fatal(err)
	}
	ref := (4*finer - fine) / 3

	var prev float64
	for i, n := range []int{g.Samples, 2 * g.Samples, 4 * g.Samples} {
		got, err := m.VerticalDisplacement(r, f, Integration{Samples: n})
		if err != nil {
			t.Fatal(err)
		}
		rel := cmplx.Abs(got-ref) / cmplx.Abs(ref)
		if i > 0 {
			order := math.Log2(prev / rel)
			t.Logf("n=%7d: relative error %.3e, order %.2f", n, rel, order)
			if order < 1.5 {
				t.Errorf("n=%d: convergence order %.2f, want second order — "+
					"a first-order rule here means the panel at k=0 is open again", n, order)
			}
		} else {
			t.Logf("n=%7d: relative error %.3e", n, rel)
			if rel > 1e-3 {
				t.Errorf("the default sampling is %.4f%% off; it should be well under a tenth of a percent",
					100*rel)
			}
		}
		prev = rel
	}
}

// The integrand at zero wavenumber is minus the static coefficient, and the
// quadrature's first panel is written on that basis rather than evaluated,
// because SurfaceResponse cannot be called at k = 0.
func TestIntegrandAtZeroWavenumber(t *testing.T) {
	m := Medium{Stack: layer.Uniform(soil.Loam())}
	const f = 40.0
	slowest, _ := m.Stack.VelocityBounds()
	kBody := 2 * math.Pi * f / float64(slowest)
	c, err := m.StaticCoefficient(f, 200*kBody)
	if err != nil {
		t.Fatal(err)
	}
	// The approach is linear in k, so the gap shrinks by ten for every decade
	// and the slope is what says the limit is -C exactly rather than -C plus
	// something small. An assertion on the gap alone could not tell those
	// apart, which is the whole question here.
	var prev float64
	for i, k := range []float64{1e-2, 1e-3, 1e-4, 1e-5} {
		u, err := m.SurfaceResponse(k, f)
		if err != nil {
			t.Fatal(err)
		}
		g := complex(k, 0)*u - c
		rel := cmplx.Abs(g+c) / cmplx.Abs(c)
		t.Logf("k=%.0e: integrand %.6e, -C %.6e, apart by %.2e", k, cmplx.Abs(g), cmplx.Abs(c), rel)
		if i > 0 {
			if shrink := prev / rel; shrink < 9 || shrink > 11 {
				t.Errorf("k=%g: the gap from -C shrank by %.2f per decade, want ten — "+
					"the limit is not -C", k, shrink)
			}
		}
		prev = rel
	}
}

func TestRejectsBadArguments(t *testing.T) {
	m := Medium{Stack: layer.Uniform(soil.Loam())}
	if _, err := m.SurfaceResponse(0, 10); err == nil {
		t.Error("expected an error for zero wavenumber")
	}
	if _, err := m.SurfaceResponse(1, 0); err == nil {
		t.Error("expected an error for zero frequency")
	}
	if _, err := m.VerticalDisplacement(0, 10, Integration{}); err == nil {
		t.Error("expected an error for zero range")
	}
	if _, err := m.VerticalDisplacement(10, 0, Integration{}); err == nil {
		t.Error("expected an error for zero frequency")
	}
	bad := Medium{Stack: layer.Stack{{Vp: 100, Vs: 200, Density: 1700}}}
	if _, err := bad.SurfaceResponse(1, 10); err == nil {
		t.Error("expected an error for an unphysical medium")
	}
}

func BenchmarkSurfaceResponse(b *testing.B) {
	m := Medium{Stack: layer.Stack{
		{Thickness: 3, Vp: 400, Vs: 160, Density: 1600},
		{Thickness: 12, Vp: 900, Vs: 400, Density: 2000},
		{Vp: 2200, Vs: 1100, Density: 2400},
	}}
	for b.Loop() {
		if _, err := m.SurfaceResponse(0.5, 20); err != nil {
			b.Fatal(err)
		}
	}
}
