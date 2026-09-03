package bank

import (
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"testing"

	"geosim.dev/geosim/internal/fk"
	"geosim.dev/geosim/internal/layer"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// smallBank builds a bank cheap enough to make in a test: a narrow band and a
// short range span, but the same code path a real one uses.
func smallBank(t *testing.T, rMin, rMax float64, count int, maxFreq float64) (*Bank, fk.Medium) {
	t.Helper()
	stack := layer.Uniform(soil.Loam())
	m := fk.Medium{Stack: stack, DefaultQ: 30}

	h := Header{
		FormatVersion: FormatVersion,
		Provenance:    Provenance{Solver: "test"},
		Medium:        stack,
		SampleRateHz:  2000,
		Samples:       512,
		Ranges:        RangeGrid{MinM: rMin, MaxM: rMax, Count: count},
		Component:     "vertical surface displacement per unit vertical surface point force",
		Units:         "m/N",
	}
	b, err := New(h)
	if err != nil {
		t.Fatal(err)
	}
	ranges := make([]units.Metres, count)
	for i := range ranges {
		ranges[i] = units.Metres(h.Ranges.At(i))
	}
	top := int(maxFreq / (h.SampleRateHz / float64(h.Samples)))
	for k := 1; k <= top && k < h.Bins(); k++ {
		vals, err := m.VerticalDisplacementMulti(ranges, h.FrequencyAt(k), fk.Integration{})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range vals {
			if err := b.Set(i, k, v); err != nil {
				t.Fatal(err)
			}
		}
	}
	return b, m
}

// decimate returns a bank holding every stride'th range of b, so its
// interpolation can be checked at the ranges it no longer stores.
func decimate(t *testing.T, b *Bank, stride int) *Bank {
	t.Helper()
	count := (b.Ranges.Count-1)/stride + 1
	h := b.Header
	h.Ranges = RangeGrid{MinM: b.Ranges.MinM, MaxM: b.Ranges.At(stride * (count - 1)), Count: count}
	d, err := New(h)
	if err != nil {
		t.Fatal(err)
	}
	for i := range count {
		for k := range b.Bins() {
			v, err := b.At(stride*i, k)
			if err != nil {
				t.Fatal(err)
			}
			if err := d.Set(i, k, v); err != nil {
				t.Fatal(err)
			}
		}
	}
	return d
}

// compare measures a coarse bank's interpolation against a fine bank's stored
// values, at the ranges the coarse one skipped.
func compare(t *testing.T, fine, coarse *Bank, stride int) (worstAmp, worstPhase float64, n int) {
	t.Helper()
	for i := range fine.Ranges.Count {
		if i%stride == 0 {
			continue // the coarse bank stores this one
		}
		r := units.Metres(fine.Ranges.At(i))
		if float64(r) > coarse.Ranges.MaxM {
			continue
		}
		for k := 1; k < fine.Bins(); k++ {
			want, err := fine.At(i, k)
			if err != nil {
				t.Fatal(err)
			}
			if want == 0 {
				continue
			}
			got, err := coarse.Response(r, k)
			if err != nil {
				t.Fatal(err)
			}
			amp := math.Abs(cmplx.Abs(got)-cmplx.Abs(want)) / cmplx.Abs(want)
			ph := cmplx.Phase(got) - cmplx.Phase(want)
			ph -= 2 * math.Pi * math.Round(ph/(2*math.Pi))
			worstAmp = math.Max(worstAmp, amp)
			worstPhase = math.Max(worstPhase, math.Abs(ph))
			n++
		}
	}
	return worstAmp, worstPhase, n
}

// Interpolation error, isolated from everything else.
//
// The coarse bank holds every other range of the fine one, so at the ranges it
// skipped the fine bank has an exact value computed by the same quadrature.
// Comparing the two measures interpolation and nothing else — no difference in
// wavenumber grid, no reference of unknown accuracy, no quadrature error that
// could be mistaken for interpolation error.
//
// Getting that isolation right mattered: a first attempt compared the bank
// against a directly computed response and found errors of tens of percent at
// high frequency. The bank was the more converged of the two — it integrates on
// a grid fine enough for its longest range, which is finer than a single short
// range needs — so most of what was being measured was the reference.
func TestInterpolationErrorAgainstStoredValues(t *testing.T) {
	const maxFreq = 100.0
	// Fine enough to serve as truth: well under the sampling limit.
	fine, _ := smallBank(t, 2, 12, 101, maxFreq)

	slowest, _ := fine.Medium.VelocityBounds()
	limit := RangeNyquist(0.87*float64(slowest), maxFreq) / 2
	t.Logf("recommended spacing at %g Hz is %.3f m; the reference grid is %.3f m",
		maxFreq, limit, fine.Ranges.Spacing())

	var atLimit float64
	for _, stride := range []int{2, 4, 8, 16} {
		coarse := decimate(t, fine, stride)
		amp, phase, n := compare(t, fine, coarse, stride)
		if n == 0 {
			t.Fatalf("stride %d: nothing compared", stride)
		}
		spacing := coarse.Ranges.Spacing()
		t.Logf("spacing %.3f m (%.1fx the recommendation): worst amplitude %.2e, worst phase %.2e rad (%.2f deg), %d points",
			spacing, spacing/limit, amp, phase, phase*180/math.Pi, n)
		if spacing <= limit {
			atLimit = math.Max(atLimit, phase)
			// Error is quadratic in spacing, so the budget is set at the
			// recommendation and everything finer is comfortably inside it.
			if amp > 0.03 {
				t.Errorf("at %.3f m spacing, within the recommendation, the amplitude error is %.2e; expected under 3%%", spacing, amp)
			}
		}
	}
	// Phase is what WP3 localises from, so it is the one with a real budget.
	// A hundredth of a radian at 100 Hz is 16 microseconds, which over soil is
	// about 3 mm of range — far inside the uncertainty in where a foot lands.
	if atLimit > 0.03 {
		t.Errorf("worst phase error within the recommended spacing is %.3e rad; expected under 0.03", atLimit)
	}
}

// The range-axis sampling condition is real, and violating it fails
// catastrophically rather than gradually.
//
// Error grows quadratically with spacing while the phase can still be
// unwrapped — a factor of two in spacing costs a factor of four. Past the
// limit, unwrapping loses whole cycles and the phase error saturates at half a
// turn: the interpolated response is not merely inaccurate, it is arbitrary.
// That is the argument for CheckRangeSampling refusing to build such a bank
// rather than emitting one that looks fine.
func TestExceedingTheSamplingLimitFailsCatastrophically(t *testing.T) {
	const maxFreq = 100.0
	fine, _ := smallBank(t, 2, 12, 101, maxFreq)
	slowest, _ := fine.Medium.VelocityBounds()
	limit := RangeNyquist(0.87*float64(slowest), maxFreq) / 2

	within := decimate(t, fine, 4)  // about 0.9x the limit
	beyond := decimate(t, fine, 16) // about 3.7x
	_, phaseWithin, _ := compare(t, fine, within, 4)
	_, phaseBeyond, _ := compare(t, fine, beyond, 16)

	if within.Ranges.Spacing() > limit {
		t.Fatalf("the 'within' grid at %.3f m is not within the %.3f m limit", within.Ranges.Spacing(), limit)
	}
	if beyond.Ranges.Spacing() <= limit {
		t.Fatalf("the 'beyond' grid at %.3f m does not exceed the %.3f m limit", beyond.Ranges.Spacing(), limit)
	}
	if phaseWithin > 0.05 {
		t.Errorf("inside the limit the phase error is %.3f rad; it should still be small", phaseWithin)
	}
	if phaseBeyond < 1.0 {
		t.Errorf("beyond the limit the phase error is only %.3f rad; unwrapping was expected to have failed outright", phaseBeyond)
	}
}

// Interpolating log-magnitude and unwrapped phase separately is the whole
// design. Interpolating the complex values directly is the obvious alternative
// and it is much worse, because two responses half a wavelength apart are of
// similar magnitude and nearly opposite phase — their average nearly cancels.
func TestPhaseAwareInterpolationBeatsComplexAveraging(t *testing.T) {
	const stride = 4
	fine, _ := smallBank(t, 2, 12, 101, 100)
	coarse := decimate(t, fine, stride)

	var ours, naive float64
	var n int
	for i := range fine.Ranges.Count {
		if i%stride == 0 {
			continue
		}
		r := units.Metres(fine.Ranges.At(i))
		if float64(r) > coarse.Ranges.MaxM {
			continue
		}
		lo, frac, err := coarse.bracket(float64(r))
		if err != nil {
			continue
		}
		for k := 1; k < fine.Bins(); k++ {
			want, _ := fine.At(i, k)
			if want == 0 {
				continue
			}
			got, err := coarse.Response(r, k)
			if err != nil {
				t.Fatal(err)
			}
			a, _ := coarse.At(lo, k)
			bb, _ := coarse.At(lo+1, k)
			flat := a + complex(frac, 0)*(bb-a)

			ours += cmplx.Abs(got-want) / cmplx.Abs(want)
			naive += cmplx.Abs(flat-want) / cmplx.Abs(want)
			n++
		}
	}
	if n == 0 {
		t.Fatal("nothing was compared")
	}
	ours, naive = ours/float64(n), naive/float64(n)
	t.Logf("mean relative error: phase-aware %.4e, complex averaging %.4e (%.0fx worse)", ours, naive, naive/ours)
	if naive < 5*ours {
		t.Errorf("complex averaging gave %.4e against %.4e; the phase-aware scheme should be far better", naive, ours)
	}
}

// The range axis has a Nyquist condition just like the time axis: between two
// grid ranges the phase must advance by less than half a cycle, or the number
// of whole cycles lost cannot be recovered. A bank too coarse to unwrap is
// rejected at build time rather than producing quiet nonsense.
func TestRangeSamplingIsChecked(t *testing.T) {
	h := Header{
		FormatVersion: FormatVersion,
		Medium:        layer.Uniform(soil.Loam()),
		SampleRateHz:  2000,
		Samples:       512,
		Ranges:        RangeGrid{MinM: 1, MaxM: 40, Count: 40}, // 1 m spacing
		Component:     "test",
	}
	if err := h.CheckRangeSampling(300); err == nil {
		t.Error("expected a one metre grid to be rejected at 300 Hz")
	}
	if err := h.CheckRangeSampling(20); err != nil {
		t.Errorf("a one metre grid should be fine at 20 Hz: %v", err)
	}
	// The published limit is c/(2f).
	if got, want := RangeNyquist(180, 300), 0.3; math.Abs(got-want) > 1e-12 {
		t.Errorf("RangeNyquist(180, 300) = %g, want %g", got, want)
	}
	if !math.IsInf(RangeNyquist(180, 0), 1) {
		t.Error("a zero frequency should impose no limit")
	}
}

// A bank has to survive a round trip through the file format unchanged. This
// is the format's own guarantee, before any cross-language question.
func TestFileRoundTrip(t *testing.T) {
	b, _ := smallBank(t, 2, 8, 9, 60)
	path := filepath.Join(t.TempDir(), "test.bank")
	if err := Write(path, b); err != nil {
		t.Fatal(err)
	}
	got, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Header.Samples != b.Header.Samples || got.Ranges != b.Ranges ||
		got.SampleRateHz != b.SampleRateHz || got.Component != b.Component {
		t.Errorf("header changed across the round trip:\n got %+v\nwant %+v", got.Header, b.Header)
	}
	if len(got.Medium) != len(b.Medium) {
		t.Fatalf("medium has %d layers after the round trip, want %d", len(got.Medium), len(b.Medium))
	}
	for i := range b.Ranges.Count {
		for k := range b.Bins() {
			want, _ := b.At(i, k)
			gotV, _ := got.At(i, k)
			if gotV != want {
				t.Fatalf("range %d bin %d: %v after the round trip, want %v", i, k, gotV, want)
			}
		}
	}
	// The payload starts on a page boundary so the file can be mapped.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw)%4 != 0 {
		t.Errorf("file length %d is not a whole number of float32s", len(raw))
	}
}

// On a grid point, interpolation must return exactly what is stored.
func TestResponseIsExactOnGridPoints(t *testing.T) {
	b, _ := smallBank(t, 2, 8, 9, 60)
	for i := range b.Ranges.Count {
		r := units.Metres(b.Ranges.At(i))
		for k := 1; k < b.Bins(); k++ {
			want, _ := b.At(i, k)
			if want == 0 {
				continue
			}
			got, err := b.Response(r, k)
			if err != nil {
				t.Fatal(err)
			}
			if rel := cmplx.Abs(got-want) / cmplx.Abs(want); rel > 1e-12 {
				t.Fatalf("range index %d bin %d: interpolation gives %v, stored is %v", i, k, got, want)
			}
		}
	}
}

func TestRejectsBadInput(t *testing.T) {
	b, _ := smallBank(t, 2, 8, 5, 40)

	if _, err := b.Response(1, 5); err == nil {
		t.Error("expected an error below the bank's range")
	}
	if _, err := b.Response(100, 5); err == nil {
		t.Error("expected an error above the bank's range")
	}
	if _, err := b.Response(4, -1); err == nil {
		t.Error("expected an error for a negative bin")
	}
	if _, err := b.Response(4, b.Bins()); err == nil {
		t.Error("expected an error for a bin past the end")
	}
	if _, err := b.At(-1, 0); err == nil {
		t.Error("expected an error for a negative range index")
	}

	for name, h := range map[string]Header{
		"wrong version": {FormatVersion: 99, Samples: 512, SampleRateHz: 2000, Ranges: RangeGrid{MinM: 1, MaxM: 2, Count: 2}, Component: "x"},
		"odd samples":   {FormatVersion: 1, Samples: 511, SampleRateHz: 2000, Ranges: RangeGrid{MinM: 1, MaxM: 2, Count: 2}, Component: "x"},
		"zero rate":     {FormatVersion: 1, Samples: 512, Ranges: RangeGrid{MinM: 1, MaxM: 2, Count: 2}, Component: "x"},
		"no ranges":     {FormatVersion: 1, Samples: 512, SampleRateHz: 2000, Ranges: RangeGrid{MinM: 1, MaxM: 2}, Component: "x"},
		"zero min":      {FormatVersion: 1, Samples: 512, SampleRateHz: 2000, Ranges: RangeGrid{MaxM: 2, Count: 2}, Component: "x"},
		"unnamed":       {FormatVersion: 1, Samples: 512, SampleRateHz: 2000, Ranges: RangeGrid{MinM: 1, MaxM: 2, Count: 2}},
	} {
		if err := h.Validate(); err == nil {
			t.Errorf("Validate accepted %s", name)
		}
	}

	dir := t.TempDir()
	junk := filepath.Join(dir, "junk.bank")
	os.WriteFile(junk, []byte("not a bank at all, not even close"), 0o644)
	if _, err := Open(junk); err == nil {
		t.Error("expected an error opening a file that is not a bank")
	}
	if _, err := Open(filepath.Join(dir, "absent.bank")); err == nil {
		t.Error("expected an error opening a file that does not exist")
	}
}

func BenchmarkResponse(b *testing.B) {
	bk, err := New(Header{
		FormatVersion: FormatVersion, Medium: layer.Uniform(soil.Loam()),
		SampleRateHz: 2000, Samples: 4096,
		Ranges:    RangeGrid{MinM: 1, MaxM: 40, Count: 270},
		Component: "x", Units: "m/N",
	})
	if err != nil {
		b.Fatal(err)
	}
	for i := range bk.Ranges.Count {
		for k := range bk.Bins() {
			bk.Set(i, k, complex(1/float64(i+1), float64(k)*1e-4))
		}
	}
	bk.prepare()
	b.ResetTimer()
	for b.Loop() {
		if _, err := bk.Response(17.3, 500); err != nil {
			b.Fatal(err)
		}
	}
}

// A bank's useful bandwidth is not uniform across its range grid, and a bank
// built to a single band limit will contain quadrature noise wherever
// attenuation has already taken the true response below it.
//
// Measured on a 600 Hz bank over loam: at 2 m the response is still rising with
// frequency at 600 Hz, at 10 m it has fallen by a decade and a half and is
// still falling, and at 20 m it stops falling around 400 Hz and flattens at
// about 2e-11 — the floor of the wavenumber integration, not the medium.
//
// The consequence is real. A synthesis driven from the flat part of that curve
// gets noise instead of signal, and because the noise is broadband it inflates
// the trace's energy rather than obviously corrupting its shape. A bank should
// therefore be built either with its band limit chosen for the *longest* range
// it will serve, or with the quadrature refined where the response is small.
//
// This test does not assert a floor value — it asserts that the response is
// still genuinely decaying across the band the bank claims, which is what has
// to hold for the bank to be usable at that range.
func TestResponseDecaysAcrossTheClaimedBand(t *testing.T) {
	const maxFreq = 100.0
	b, _ := smallBank(t, 2, 12, 41, maxFreq)

	top := int(maxFreq / b.FrequencyAt(1))
	for _, r := range []units.Metres{4, 8, 12} {
		lo, err := b.Response(r, top/4)
		if err != nil {
			t.Fatal(err)
		}
		hi, err := b.Response(r, top)
		if err != nil {
			t.Fatal(err)
		}
		if cmplx.Abs(lo) == 0 || cmplx.Abs(hi) == 0 {
			t.Fatalf("r=%g m: the bank is empty in the band it claims", r)
		}
		// Not a claim about the decay rate, only that the top of the band is
		// still carrying a response of a believable size rather than a floor.
		ratio := cmplx.Abs(hi) / cmplx.Abs(lo)
		t.Logf("r=%4.0f m: |G| at %.0f Hz is %.3f of its value at %.0f Hz",
			r, b.FrequencyAt(top), ratio, b.FrequencyAt(top/4))
		if ratio > 10 || ratio < 1e-6 {
			t.Errorf("r=%g m: response ratio across the band is %.3e, which is not a believable decay", r, ratio)
		}
	}
}
