package dsp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// V10: Parseval. Energy in the time domain equals energy in the frequency
// domain. This is the check that the FFT wrapper's normalisation is right —
// gonum's inverse is unnormalised, and getting that wrong scales every
// synthesised waveform by the FFT length without changing its shape, which is
// exactly the kind of error that survives a visual inspection.
func TestParseval(t *testing.T) {
	for _, n := range []int{16, 64, 256, 1000, 1024} {
		rng := rand.New(rand.NewPCG(1, uint64(n)))
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		time := Energy(x)
		freq := SpectralEnergy(RFFT(x), n)
		if rel := math.Abs(time-freq) / time; rel > 1e-12 {
			t.Errorf("n=%d: time energy %g, spectral energy %g, rel err %g", n, time, freq, rel)
		}
	}
}

// V10: the forward transform inverts.
func TestRFFTRoundTrip(t *testing.T) {
	for _, n := range []int{8, 100, 512} {
		rng := rand.New(rand.NewPCG(2, uint64(n)))
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		got := IRFFT(RFFT(x), n)
		if len(got) != n {
			t.Fatalf("n=%d: round trip returned length %d", n, len(got))
		}
		for i := range x {
			if math.Abs(got[i]-x[i]) > 1e-12 {
				t.Fatalf("n=%d sample %d: got %g, want %g", n, i, got[i], x[i])
			}
		}
	}
}

// V10: the FFT convolution agrees with the definition. Guards the
// zero-padding — a circular convolution passes a round-trip test and fails
// this one.
func TestConvolveMatchesDirect(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, sz := range [][2]int{{1, 1}, {5, 3}, {17, 64}, {200, 137}} {
		a := make([]float64, sz[0])
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		b := make([]float64, sz[1])
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		got, want := Convolve(a, b), ConvolveDirect(a, b)
		if len(got) != len(want) {
			t.Fatalf("%v: length %d, want %d", sz, len(got), len(want))
		}
		scale := math.Sqrt(Energy(want))
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-10*scale {
				t.Fatalf("%v sample %d: got %g, want %g", sz, i, got[i], want[i])
			}
		}
	}
}

// A delta is the convolution identity, and a delayed delta is a pure shift.
// If this fails, everything downstream is time-shifted.
func TestConvolveDelta(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	d := []float64{0, 0, 1}
	got := Convolve(x, d)
	for i, want := range x {
		if math.Abs(got[i+2]-want) > 1e-12 {
			t.Errorf("shifted sample %d: got %g, want %g", i, got[i+2], want)
		}
	}
	for i := 0; i < 2; i++ {
		if math.Abs(got[i]) > 1e-12 {
			t.Errorf("sample %d before the delay: got %g, want 0", i, got[i])
		}
	}
}

func TestNextPow2(t *testing.T) {
	for _, c := range [][2]int{{0, 1}, {1, 1}, {2, 2}, {3, 4}, {1023, 1024}, {1024, 1024}, {1025, 2048}} {
		if got := NextPow2(c[0]); got != c[1] {
			t.Errorf("NextPow2(%d) = %d, want %d", c[0], got, c[1])
		}
	}
}

// Resampling must preserve a tone well below both Nyquist frequencies. The
// tolerance is loose at the edges, where the Lanczos window runs off the end
// of the data, so the interior is what is checked.
func TestResampleTone(t *testing.T) {
	const (
		in   = 2000.0
		out  = 1000.0
		freq = 40.0
		n    = 2000
	)
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2 * math.Pi * freq * float64(i) / in)
	}
	got, err := Resample(x, in, out, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != n/2 {
		t.Fatalf("length %d, want %d", len(got), n/2)
	}
	for i := 50; i < len(got)-50; i++ {
		want := math.Sin(2 * math.Pi * freq * float64(i) / out)
		if math.Abs(got[i]-want) > 5e-3 {
			t.Fatalf("sample %d: got %g, want %g", i, got[i], want)
		}
	}
}

func TestResampleRejectsBadArgs(t *testing.T) {
	if _, err := Resample([]float64{1}, 0, 1, 4); err == nil {
		t.Error("expected an error for a zero input rate")
	}
	if _, err := Resample([]float64{1}, 1, 1, 0); err == nil {
		t.Error("expected an error for a zero window half-width")
	}
}

func BenchmarkConvolve(b *testing.B) {
	rng := rand.New(rand.NewPCG(5, 6))
	src := make([]float64, 2048)
	for i := range src {
		src[i] = rng.NormFloat64()
	}
	gf := make([]float64, 4096)
	for i := range gf {
		gf[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for b.Loop() {
		Convolve(src, gf)
	}
}
