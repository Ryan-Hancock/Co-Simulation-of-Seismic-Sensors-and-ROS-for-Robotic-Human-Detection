package hier

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

func smallOptions(ranges ...units.Metres) Options {
	return Options{
		Ranges: ranges, Band: 40, Rate: 200, Samples: 512, Q: 30,
		ReferenceRefinement: 4,
	}
}

func mustWavenumber(t *testing.T, st layer.Stack, q float64, refine int, name string) Level {
	t.Helper()
	lv, err := Wavenumber(st, q, refine, name, "test")
	if err != nil {
		t.Fatal(err)
	}
	return lv
}

// A level scored against itself must be exactly zero. It is the cheapest
// possible check that the comparison is comparing, and it would catch an
// off-by-one between the reference's bins and a level's.
func TestALevelAgainstItselfIsExact(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	opt := smallOptions(4, 10)
	lv := mustWavenumber(t, st, opt.Q, 1, "self")
	rep, err := Compare(lv, []Level{lv}, opt)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range rep.Scores {
		if s.SpectralMax != 0 || s.TraceRMS != 0 || s.PeakRatio != 1 {
			t.Errorf("r=%g m: a level scored against itself gave max %g, rms %g, peak %g",
				s.Range, s.SpectralMax, s.TraceRMS, s.PeakRatio)
		}
	}
}

// The reference has to be converged well below the errors it is used to
// measure, or the table reports the reference's own error as everyone else's.
//
// That is not hypothetical here: slice 4's first comparison did exactly this
// and the residual it produced sat still under refinement, which reads like a
// wrong answer somewhere else entirely.
func TestReferenceIsConverged(t *testing.T) {
	st := layer.Stack{
		{Thickness: 2, Vp: 350, Vs: 140, Density: 1600},
		{Vp: 800, Vs: 320, Density: 2000},
	}
	opt := smallOptions(3, 12)
	ref := mustWavenumber(t, st, opt.Q, 2*opt.refinement(), "twice as fine")
	rep, err := Compare(ref, []Level{mustWavenumber(t, st, opt.Q, opt.refinement(), "reference")}, opt)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range rep.Scores {
		t.Logf("r=%4.1f m: the reference differs from twice its sampling by %.2e (median %.2e)",
			s.Range, s.SpectralMax, s.SpectralMedian)
		if s.SpectralMax > 1e-3 {
			t.Errorf("r=%g m: the reference moves by %.3f%% when refined; it is not converged enough "+
				"to measure anything at the tenth-of-a-percent level", s.Range, 100*s.SpectralMax)
		}
	}
}

// The far-field closed form is right where it claims to be and wrong where it
// does not, and the table has to show both.
//
// On a homogeneous half-space at long range it should land close to the full
// wavenumber integration; at short range the near-field terms it omits are the
// signal, and it should not. A comparison that showed the level failing
// everywhere, or nowhere, would be measuring itself.
func TestAnalyticLevelIsRightWhereItClaimsToBe(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	opt := smallOptions(1, 20)
	// Scored from 15 Hz up, where 20 m is more than two wavelengths and the
	// asymptotic form is entitled to be right. Below that it is not, at any
	// range, and including it would measure the band rather than the model.
	opt.BandLow = 15
	ref := mustWavenumber(t, st, opt.Q, opt.refinement(), "reference")
	rep, err := Compare(ref, []Level{Analytic(st, opt.Q)}, opt)
	if err != nil {
		t.Fatal(err)
	}
	near, far := rep.Scores[0], rep.Scores[1]
	t.Logf("far-field model: %.1f%% median at %g m, %.1f%% median at %g m",
		100*near.SpectralMedian, near.Range, 100*far.SpectralMedian, far.Range)
	if far.SpectralMedian > 0.2 {
		t.Errorf("at %g m and above %g Hz the far-field model is %.1f%% from the integration; "+
			"it should be close where it claims to work", far.Range, opt.BandLow, 100*far.SpectralMedian)
	}
	if near.SpectralMedian < 3*far.SpectralMedian {
		t.Errorf("the far-field model is no worse at %g m (%.3f) than at %g m (%.3f); "+
			"the near field it omits should dominate at short range",
			near.Range, near.SpectralMedian, far.Range, far.SpectralMedian)
	}
}

// Layering is the thing the closed form cannot represent, and the table's
// headline number is how much that costs.
func TestAnalyticLevelFailsOnALayeredSite(t *testing.T) {
	st := layer.Stack{
		{Thickness: 2, Vp: 350, Vs: 140, Density: 1600},
		{Vp: 800, Vs: 320, Density: 2000},
	}
	opt := smallOptions(5, 15)
	ref := mustWavenumber(t, st, opt.Q, opt.refinement(), "reference")
	rep, err := Compare(ref, []Level{Analytic(st, opt.Q)}, opt)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range rep.Scores {
		t.Logf("r=%4.1f m: median %.1f%%, trace rms %.1f%%, peak ratio %.3f",
			s.Range, 100*s.SpectralMedian, 100*s.TraceRMS, s.PeakRatio)
		if s.SpectralMedian < 0.2 {
			t.Errorf("r=%g m: the top-layer half-space is only %.1f%% off a two-layer site; "+
				"either the contrast is too weak to be testing anything or the levels are not "+
				"modelling different media", s.Range, 100*s.SpectralMedian)
		}
	}
}

// A bank is exact at its nodes, so scoring one there measures nothing. The grid
// is deliberately offset by half a step for that reason, and this pins it:
// remove the offset and the reported interpolation error collapses to zero at
// every scored range.
func TestBankIsScoredBetweenItsNodes(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	opt := smallOptions(4, 9, 14)
	lv, b, err := Banked(st, opt)
	if err != nil {
		t.Fatal(err)
	}
	step := b.Ranges.Spacing()
	for _, r := range opt.Ranges {
		off := math.Mod(float64(r)-b.Ranges.MinM, step) / step
		t.Logf("r=%g m sits %.2f of the way through a cell", r, off)
		if off < 0.05 || off > 0.95 {
			t.Errorf("r=%g m is effectively on a bank node (%.3f of a cell); "+
				"the level would be scored where its interpolation error does not exist", r, off)
		}
	}
	ref := mustWavenumber(t, st, opt.Q, opt.refinement(), "reference")
	rep, err := Compare(ref, []Level{lv}, opt)
	if err != nil {
		t.Fatal(err)
	}
	var worst float64
	for _, s := range rep.Scores {
		worst = math.Max(worst, s.SpectralMedian)
		t.Logf("r=%4.1f m: bank median %.3f%%, max %.3f%% at %.0f Hz, trace rms %.3f%%",
			s.Range, 100*s.SpectralMedian, 100*s.SpectralMax, s.WorstAt, 100*s.TraceRMS)
	}
	if worst == 0 {
		t.Error("the bank level reported no error at any range, which means it was scored on its nodes")
	}
	if worst > 0.05 {
		t.Errorf("bank interpolation is %.2f%% off at its worst range, well beyond what "+
			"the sampling limit should allow", 100*worst)
	}
}

// A bank has to cost something to build and almost nothing to use, or there is
// no reason for it to exist.
func TestBankTradesBuildTimeForLookupTime(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	opt := smallOptions(4, 12)
	lv, _, err := Banked(st, opt)
	if err != nil {
		t.Fatal(err)
	}
	full := mustWavenumber(t, st, opt.Q, 1, "wavenumber")
	ref := mustWavenumber(t, st, opt.Q, opt.refinement(), "reference")
	rep, err := Compare(ref, []Level{full, lv}, opt)
	if err != nil {
		t.Fatal(err)
	}
	var direct, lookup float64
	for _, s := range rep.Scores {
		switch s.Level {
		case "wavenumber":
			direct = float64(s.PerResponse)
		case lv.Name:
			lookup = float64(s.PerResponse)
		}
	}
	t.Logf("build %v, then %.0f ns a lookup against %.0f µs an integration (%.0fx)",
		lv.Setup.Round(1e6), lookup, direct/1e3, direct/lookup)
	if lookup <= 0 || direct/lookup < 1e3 {
		t.Errorf("a lookup is %.0f ns and an integration %.0f ns; the bank is not buying enough "+
			"to justify its build cost", lookup, direct)
	}
	if lv.Setup <= 0 {
		t.Error("the bank build was not charged to the level")
	}
}

// The inverse transform is periodic, so a response still ringing at the end of
// the window wraps onto the arrival at the start, where it is indistinguishable
// from a level getting the arrival wrong.
func TestSuggestedGridOutlastsTheCoda(t *testing.T) {
	for _, c := range []struct {
		band     float64
		slowest  units.SpeedMPS
		furthest units.Metres
	}{{40, 200, 20}, {150, 140, 20}, {600, 120, 40}} {
		rate, n := SuggestGrid(c.band, c.slowest, c.furthest)
		if rate <= 2*c.band {
			t.Errorf("rate %g Hz is not above Nyquist for a %g Hz band", rate, c.band)
		}
		window := float64(n) / rate
		arrival := float64(c.furthest) / (0.87 * float64(c.slowest))
		t.Logf("band %g Hz: %g Hz for %d samples is a %.3f s window against a %.3f s arrival",
			c.band, rate, n, window, arrival)
		if window < 3*arrival {
			t.Errorf("a %.3f s window leaves no room after a %.3f s arrival", window, arrival)
		}
	}
}

func TestRejectsBadOptions(t *testing.T) {
	st := layer.Uniform(soil.Loam())
	ref := mustWavenumber(t, st, 30, 1, "ref")
	for name, opt := range map[string]Options{
		"no ranges":          {Band: 40, Rate: 200, Samples: 512},
		"no band":            {Ranges: []units.Metres{4}, Rate: 200, Samples: 512},
		"band above nyquist": {Ranges: []units.Metres{4}, Band: 150, Rate: 200, Samples: 512},
		"no grid":            {Ranges: []units.Metres{4}, Band: 40},
	} {
		if _, err := Compare(ref, nil, opt); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
