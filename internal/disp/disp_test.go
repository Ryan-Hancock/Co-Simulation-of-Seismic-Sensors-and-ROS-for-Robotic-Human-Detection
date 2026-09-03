package disp

import (
	"math"
	"path/filepath"
	"strconv"
	"testing"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
)

func goldens(t *testing.T) []layer.Golden {
	t.Helper()
	paths, err := filepath.Glob("../../testdata/dispersion/*.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) < 4 {
		t.Fatalf("found %d golden files; expected the full set from py/oracles/dispersion.py", len(paths))
	}
	var out []layer.Golden
	for _, p := range paths {
		g, err := layer.LoadGolden(p)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, g)
	}
	return out
}

// V3: the dispersion curves, against disba.
//
// This is the test the whole slice turns on. A propagator with a sign error, a
// compound matrix with a transposed index, or a root finder that has drifted
// onto the wrong mode all produce curves that are smooth, monotone and
// physically shaped — there is nothing in the output to say they are wrong.
// The only defence is an independently written implementation, and disba is
// that. The comparison is against committed golden files, so the Go tests
// carry no Python dependency: the seam is crossed once, when the file is
// generated.
func TestDispersionMatchesDisba(t *testing.T) {
	for _, g := range goldens(t) {
		t.Run(g.Name, func(t *testing.T) {
			for modeStr, want := range g.Modes {
				mode, err := strconv.Atoi(modeStr)
				if err != nil {
					t.Fatal(err)
				}
				var checked int
				var worst float64
				for i, f := range want.Frequency {
					got, err := Mode(g.Layers, f, mode, Search{})
					if err != nil {
						// disba found the mode here and we did not. That is a
						// miss, and a miss is the failure that matters.
						t.Errorf("mode %d at %.3f Hz: disba has %.2f m/s, this solver found none (%v)",
							mode, f, want.PhaseVelocity[i], err)
						continue
					}
					rel := (float64(got) - want.PhaseVelocity[i]) / want.PhaseVelocity[i]
					if math.Abs(rel) > math.Abs(worst) {
						worst = rel
					}
					if math.Abs(rel) > 1e-4 {
						t.Errorf("mode %d at %.3f Hz: %.4f m/s, disba gives %.4f (rel %.2e)",
							mode, f, got, want.PhaseVelocity[i], rel)
					}
					checked++
				}
				if checked == 0 {
					t.Errorf("mode %d: nothing was checked", mode)
				}
				t.Logf("mode %d: %d points, worst relative error %.2e", mode, checked, worst)
			}
		})
	}
}

// Modes must come back slowest first and all distinct. The ordering is what
// gives "mode 0" its meaning, and duplicates would mean the scan is finding
// the same root twice — which is how a higher mode gets reported as the
// fundamental.
func TestModesAreOrderedAndDistinct(t *testing.T) {
	for _, g := range goldens(t) {
		t.Run(g.Name, func(t *testing.T) {
			lo, hi := Bounds(g.Layers)
			for _, f := range []float64{5, 20, 60, 110} {
				all, err := Modes(g.Layers, f, Search{})
				if err != nil {
					t.Fatal(err)
				}
				for i, c := range all {
					if float64(c) <= lo || float64(c) >= hi {
						t.Errorf("%g Hz: mode %d at %g m/s is outside the search bounds [%g, %g]", f, i, c, lo, hi)
					}
					if i > 0 && c <= all[i-1] {
						t.Errorf("%g Hz: mode %d (%g) is not above mode %d (%g)", f, i, c, i-1, all[i-1])
					}
				}
			}
		})
	}
}

// A layered medium disperses and a homogeneous one does not. Both halves
// matter: the first is the point of the slice, the second is the check that
// dispersion is coming from the layering rather than from an artefact of the
// solver.
func TestLayeringCausesDispersion(t *testing.T) {
	t.Run("homogeneous does not disperse", func(t *testing.T) {
		h := soil.Loam()
		want, err := h.RayleighVelocity()
		if err != nil {
			t.Fatal(err)
		}
		st := layer.Uniform(h)
		for _, f := range []float64{2, 20, 120} {
			got, err := Mode(st, f, 0, Search{})
			if err != nil {
				t.Fatal(err)
			}
			if rel := math.Abs(float64(got-want)) / float64(want); rel > 1e-6 {
				t.Errorf("%g Hz: %g m/s, want the homogeneous Rayleigh velocity %g", f, got, want)
			}
		}
	})

	t.Run("soft over stiff is normally dispersive", func(t *testing.T) {
		// Long waves reach the stiff half-space and travel fast; short waves
		// stay in the slow surface layer. So phase velocity falls with
		// frequency, monotonically.
		st := layer.Stack{
			{Thickness: 3, Vp: 400, Vs: 160, Density: 1600},
			{Vp: 900, Vs: 400, Density: 2000},
		}
		var prev float64 = math.Inf(1)
		for _, f := range []float64{2, 5, 10, 20, 40, 80, 120} {
			got, err := Mode(st, f, 0, Search{})
			if err != nil {
				t.Fatal(err)
			}
			if float64(got) >= prev {
				t.Errorf("%g Hz: %g m/s did not fall below the previous %g", f, got, prev)
			}
			prev = float64(got)
		}
		// And it is bracketed by the two materials' Rayleigh velocities.
		slow, err := soil.HalfSpace{Vp: 400, Vs: 160, Density: 1600, Qs: 20}.RayleighVelocity()
		if err != nil {
			t.Fatal(err)
		}
		hiF, err := Mode(st, 120, 0, Search{})
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(hiF-slow))/float64(slow) > 0.05 {
			t.Errorf("at 120 Hz the mode is %g m/s; it should approach the surface layer's own Rayleigh velocity %g", hiF, slow)
		}
	})
}

// Higher modes have cutoff frequencies: below them they do not exist. Asking
// for one must be an error rather than a silently substituted fundamental,
// because a solver that quietly returns the wrong mode is exactly the failure
// this package is built to avoid.
func TestHigherModesHaveCutoffs(t *testing.T) {
	st := layer.Stack{
		{Thickness: 3, Vp: 400, Vs: 160, Density: 1600},
		{Vp: 900, Vs: 400, Density: 2000},
	}
	if _, err := Mode(st, 2, 1, Search{}); err == nil {
		t.Error("expected the first higher mode not to exist at 2 Hz")
	}
	if _, err := Mode(st, 100, 1, Search{}); err != nil {
		t.Errorf("expected the first higher mode to exist at 100 Hz: %v", err)
	}
	// PhaseCurve skips the frequencies where a mode has not cut in, so its
	// two slices stay aligned with each other.
	freqs := []float64{2, 5, 10, 20, 40, 80, 120}
	c, err := PhaseCurve(st, freqs, 1, Search{})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Frequency) != len(c.PhaseVelocity) {
		t.Fatalf("curve has %d frequencies and %d velocities", len(c.Frequency), len(c.PhaseVelocity))
	}
	if len(c.Frequency) == 0 || len(c.Frequency) == len(freqs) {
		t.Errorf("first higher mode exists at %d of %d frequencies; expected a cutoff inside the range", len(c.Frequency), len(freqs))
	}
}

// Group velocity is what carries energy, so it is what sets when a wave packet
// arrives. In a normally dispersive medium it lies below the phase velocity,
// and the gap is large enough to matter for arrival picking.
func TestGroupVelocityBelowPhaseWhereNormallyDispersive(t *testing.T) {
	st := layer.Stack{
		{Thickness: 3, Vp: 400, Vs: 160, Density: 1600},
		{Vp: 900, Vs: 400, Density: 2000},
	}
	var checked int
	for _, f := range []float64{10, 20, 40} {
		c, err := Mode(st, f, 0, Search{})
		if err != nil {
			t.Fatal(err)
		}
		u, err := GroupVelocity(st, f, 0, Search{})
		if err != nil {
			t.Fatal(err)
		}
		if u >= c {
			t.Errorf("%g Hz: group velocity %g is not below phase velocity %g", f, u, c)
		}
		if u <= 0 {
			t.Errorf("%g Hz: group velocity %g is not positive", f, u)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("nothing was checked")
	}

	// A homogeneous medium does not disperse, so the two must coincide.
	uni := layer.Uniform(soil.Loam())
	c, err := Mode(uni, 20, 0, Search{})
	if err != nil {
		t.Fatal(err)
	}
	u, err := GroupVelocity(uni, 20, 0, Search{})
	if err != nil {
		t.Fatal(err)
	}
	if rel := math.Abs(float64(u-c)) / float64(c); rel > 1e-4 {
		t.Errorf("homogeneous medium: group %g and phase %g differ by %g", u, c, rel)
	}
}

func TestRejectsBadArguments(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	if _, err := Modes(st, 0, Search{}); err == nil {
		t.Error("expected an error for zero frequency")
	}
	if _, err := Mode(st, 20, -1, Search{}); err == nil {
		t.Error("expected an error for a negative mode index")
	}
	if _, err := Mode(st, 20, 9, Search{}); err == nil {
		t.Error("expected an error for a mode that does not exist")
	}
	bad := layer.Stack{{Thickness: 3, Vp: 100, Vs: 200, Density: 1600}, {Vp: 900, Vs: 400, Density: 2000}}
	if _, err := Modes(bad, 20, Search{}); err == nil {
		t.Error("expected an error for an unphysical stack")
	}
}

func BenchmarkFundamentalMode(b *testing.B) {
	st := layer.Stack{
		{Thickness: 3, Vp: 400, Vs: 160, Density: 1600},
		{Thickness: 12, Vp: 900, Vs: 400, Density: 2000},
		{Vp: 2200, Vs: 1100, Density: 2400},
	}
	for b.Loop() {
		if _, err := Mode(st, 30, 0, Search{}); err != nil {
			b.Fatal(err)
		}
	}
}
