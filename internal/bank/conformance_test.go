package bank

import (
	"math"
	"math/cmplx"
	"testing"
)

// conformPattern must match py/bankfmt/conformance.py and cmd/geobank exactly.
func conformPattern(i, k int) complex128 {
	return complex(float64(i+1)*math.Pow(10, float64(k%5-2)), -float64(k+1)*0.25+float64(i))
}

// The format has two implementations, in two languages, and two
// implementations of one format drift. This reads the fixture the Python side
// wrote and checks every value.
//
// The fixture is committed rather than generated, so these tests need no Python
// at all — the seam is crossed once, when the file is produced, exactly as it
// is for the dispersion golden curves. Regenerate with:
//
//	py/.venv/bin/python py/bankfmt/conformance.py write testdata/bank/python_written.bank
//
// The payload is a pattern rather than physics on purpose. A transposed layout,
// a swapped real and imaginary part, a byte-order mistake or a float32/float64
// confusion each produce a mismatch here; against real Green's functions all
// four would produce numbers that still looked like Green's functions.
func TestReadsBankWrittenByPython(t *testing.T) {
	b, err := Open("../../testdata/bank/python_written.bank")
	if err != nil {
		t.Fatalf("opening the Python-written fixture: %v", err)
	}
	if b.Ranges.Count != 7 || b.Samples != 32 {
		t.Fatalf("fixture has %d ranges and %d samples, want 7 and 32", b.Ranges.Count, b.Samples)
	}
	if b.SampleRateHz != 2000 {
		t.Errorf("sample rate %g, want 2000", b.SampleRateHz)
	}
	if len(b.Medium) != 1 || b.Medium[0].Vs != 200 {
		t.Errorf("medium did not survive: %+v", b.Medium)
	}

	var worst float64
	for i := range b.Ranges.Count {
		for k := range b.Bins() {
			got, err := b.At(i, k)
			if err != nil {
				t.Fatal(err)
			}
			want := conformPattern(i, k)
			d := cmplx.Abs(got-want) / math.Max(cmplx.Abs(want), 1e-30)
			worst = math.Max(worst, d)
		}
	}
	// float32 on disk sets the floor.
	if worst > 1e-6 {
		t.Errorf("worst relative difference %.3e against the conformance pattern", worst)
	}
	t.Logf("read %d ranges x %d bins written by Python; worst relative difference %.3e",
		b.Ranges.Count, b.Bins(), worst)
}

// And the other direction: the fixture Go writes, which the Python side checks
// with `python py/bankfmt/conformance.py check testdata/bank/go_written.bank`.
// Verifying it here too means a change that breaks the format is caught by the
// Go tests even if nobody runs the Python half.
func TestGoWrittenFixtureMatchesThePattern(t *testing.T) {
	b, err := Open("../../testdata/bank/go_written.bank")
	if err != nil {
		t.Fatalf("opening the Go-written fixture: %v", err)
	}
	for i := range b.Ranges.Count {
		for k := range b.Bins() {
			got, _ := b.At(i, k)
			want := conformPattern(i, k)
			if d := cmplx.Abs(got-want) / math.Max(cmplx.Abs(want), 1e-30); d > 1e-6 {
				t.Fatalf("range %d bin %d: %v, want %v", i, k, got, want)
			}
		}
	}
}
