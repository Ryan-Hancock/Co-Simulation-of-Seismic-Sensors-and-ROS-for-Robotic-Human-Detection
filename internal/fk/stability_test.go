package fk

import (
	"math"
	"math/cmplx"
	"testing"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// How far up in frequency-thickness the layered solve can be trusted.
//
// V4 establishes that the *dispersion* solver survives high f*h, because
// internal/propmat uses Dunkin's compound matrices where the plain
// Thomson-Haskell recursion loses its decaying solution to rounding. This
// package does not use them. It propagates the four-component motion-stress
// vector directly, which is the formulation Dunkin's exists to replace, so V4
// says nothing about it and the two are separate code paths.
//
// The test isolates the arithmetic from the physics completely: a stack of two
// identical layers over a half-space of the same material *is* that half-space,
// so any difference between them is the layered machinery losing precision and
// nothing else. No reference solution is needed and no physics is assumed.
//
// The failure is not silent — the solve reports a singular system rather than
// returning a plausible number — but it is preceded by a band where the answer
// degrades quietly, and that band is what this measures.
func TestLayeredSolveAgreesWithTheHalfSpaceItIsMadeOf(t *testing.T) {
	h := soil.Loam()
	half := Medium{Stack: layer.Uniform(h), DefaultQ: 30}
	// The same material, cut into two layers over a half-space of itself.
	l := layer.Layer{Thickness: 2, Vp: h.Vp, Vs: h.Vs, Density: h.Density, Qs: h.Qs}
	split := Medium{Stack: layer.Stack{l, l, {Vp: h.Vp, Vs: h.Vs, Density: h.Density, Qs: h.Qs}}, DefaultQ: 30}

	const r = units.Metres(10)
	worst := 0.0
	var brokeAt float64
	for _, f := range []float64{10, 30, 100, 200, 400, 600, 800, 1000, 1400} {
		a, err := half.VerticalDisplacement(r, f, Integration{})
		if err != nil {
			t.Fatalf("%g Hz: the half-space itself failed: %v", f, err)
		}
		b, err := split.VerticalDisplacement(r, f, Integration{})
		if err != nil {
			t.Logf("%6.0f Hz: the split stack reports %v", f, err)
			if brokeAt == 0 {
				brokeAt = f
			}
			continue
		}
		rel := cmplx.Abs(a-b) / cmplx.Abs(a)
		// k*h at the body-wave wavenumber, which is what sets the size of the
		// exponentials the propagator has to carry.
		kh := 2 * math.Pi * f / float64(h.Vs) * float64(l.Thickness)
		t.Logf("%6.0f Hz (k*h ~ %5.1f): identical layers differ from the half-space by %.2e", f, kh, rel)
		worst = math.Max(worst, rel)
		if rel > 1e-6 && brokeAt == 0 {
			brokeAt = f
		}
	}
	if brokeAt > 0 {
		t.Logf("the layered path departs from its own half-space above about %g Hz "+
			"for a %g m layer", brokeAt, l.Thickness)
	}
	// Below the guard's limit the layered path must agree with its own
	// half-space to well past any other error in the model; above it, it must
	// refuse rather than answer.
	for _, f := range []float64{10, 30, 100, 200} {
		a, err := half.VerticalDisplacement(r, f, Integration{})
		if err != nil {
			t.Fatal(err)
		}
		b, err := split.VerticalDisplacement(r, f, Integration{})
		if err != nil {
			t.Fatalf("%g Hz is inside the band the guard allows and the solve refused: %v", f, err)
		}
		if rel := cmplx.Abs(a-b) / cmplx.Abs(a); rel > 1e-5 {
			t.Errorf("%g Hz: identical layers differ from the half-space by %.2e, which is "+
				"arithmetic rather than physics", f, rel)
		}
	}
	if _, err := split.VerticalDisplacement(r, 400, Integration{}); err == nil {
		t.Error("400 Hz over a 2 m layer was answered rather than refused; measured, that answer " +
			"is 132% wrong")
	}
	// The half-space itself has no layers, so nothing constrains it.
	if _, err := half.VerticalDisplacement(r, 400, Integration{}); err != nil {
		t.Errorf("the guard fired on a medium with no layers: %v", err)
	}
}

// The limit is on k times thickness, not on frequency, so a thinner layer must
// buy proportionally more bandwidth. If the guard were keyed on frequency alone
// it would be wrong for every stack but the one it was tuned on.
func TestLayerPhaseLimitScalesWithThickness(t *testing.T) {
	h := soil.Loam()
	var prev float64
	for _, thick := range []units.Metres{1, 2, 4} {
		l := layer.Layer{Thickness: thick, Vp: h.Vp, Vs: h.Vs, Density: h.Density, Qs: h.Qs}
		m := Medium{Stack: layer.Stack{l, {Vp: h.Vp, Vs: h.Vs, Density: h.Density, Qs: h.Qs}}, DefaultQ: 30}
		// Find the highest frequency it will answer at, to within a hertz.
		lo, hi := 1.0, 4000.0
		for range 40 {
			mid := 0.5 * (lo + hi)
			if m.checkLayerPhase(mid) == nil {
				lo = mid
			} else {
				hi = mid
			}
		}
		t.Logf("a %g m layer is usable to %.0f Hz", thick, lo)
		if prev > 0 {
			if ratio := prev / lo; math.Abs(ratio-2) > 0.02 {
				t.Errorf("halving the thickness changed the usable band by %.3fx, want 2", ratio)
			}
		}
		prev = lo
	}
}
