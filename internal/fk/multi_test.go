package fk

import (
	"math/cmplx"
	"testing"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// Sharing the wavenumber samples across ranges must not change the answer —
// it is the same quadrature, evaluated once instead of once per range.
func TestMultiMatchesSingleRange(t *testing.T) {
	m := Medium{Stack: layer.Uniform(soil.Loam())}
	ranges := []units.Metres{2, 5, 10, 20}
	const f = 20.0

	g, err := m.GridFor(ranges, f, Integration{})
	if err != nil {
		t.Fatal(err)
	}
	multi, err := m.VerticalDisplacementMulti(ranges, f, Integration{})
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range ranges {
		// The single-range call chooses its own grid, so match it explicitly.
		single, err := m.VerticalDisplacement(r, f, Integration{Samples: g.Samples, KMaxFactor: g.KMax / (2 * 3.141592653589793 * f / 200)})
		if err != nil {
			t.Fatal(err)
		}
		if rel := cmplx.Abs(multi[i]-single) / cmplx.Abs(single); rel > 0.05 {
			t.Errorf("r=%g m: shared-grid %v, per-range %v (rel %.4f)", r, multi[i], single, rel)
		}
	}
}

// The grid has to be chosen from the extremes of the range set: far enough out
// for the shortest range's near field, fine enough for the longest range's
// Hankel oscillation and for the Rayleigh pole.
func TestGridServesTheWholeRangeSet(t *testing.T) {
	m := Medium{Stack: layer.Uniform(soil.Loam())}
	near, err := m.GridFor([]units.Metres{1, 50}, 20, Integration{})
	if err != nil {
		t.Fatal(err)
	}
	if near.KMax < 40 {
		t.Errorf("kMax = %g; a one metre range needs the integral carried to about 40 per metre", near.KMax)
	}
	// Fine enough for the fifty metre Hankel oscillation.
	if dk := near.KMax / float64(near.Samples); dk > 2*3.141592653589793/(8*50) {
		t.Errorf("dk = %g is too coarse for a fifty metre range", dk)
	}
	if _, err := m.GridFor(nil, 20, Integration{}); err == nil {
		t.Error("expected an error for an empty range set")
	}
	if _, err := m.GridFor([]units.Metres{0}, 20, Integration{}); err == nil {
		t.Error("expected an error for a zero range")
	}
}
