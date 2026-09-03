package stream

import (
	"math"
	"math/rand/v2"
	"testing"

	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/soil"
	"geosim.dev/geosim/internal/units"
)

// V12, the reason this package exists. Chunked output must equal monolithic
// output to machine precision — not approximately, not to within a tolerance
// that would hide a small discontinuity.
//
// The failure this guards against is specific and nasty. If state is not
// carried across a boundary, every chunk edge gets a step in ground velocity.
// A step in ground velocity is what a footstep looks like, so WP3's detector
// would find them — at perfectly regular intervals, with nothing in the
// waveform to mark them as artificial.
func TestChunkedEqualsMonolithic(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	impulse := make([]float64, 700)
	for i := range impulse {
		// A decaying oscillation: the shape of a real Green's function, and
		// long enough to span many chunks.
		impulse[i] = math.Exp(-float64(i)/120) * math.Sin(float64(i)*0.21)
	}
	in := make([]float64, 4096)
	for i := range in {
		in[i] = rng.NormFloat64()
	}
	want := Monolithic(in, impulse)

	for _, chunk := range []int{1, 2, 7, 16, 20, 64, 128, 512, 1024} {
		c, err := NewConvolver(impulse, chunk)
		if err != nil {
			t.Fatal(err)
		}
		got := c.ProcessAll(in)

		var scale float64
		for _, v := range want {
			scale = math.Max(scale, math.Abs(v))
		}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-9*scale {
				t.Fatalf("chunk=%d: sample %d is %g, monolithic gives %g (diff %g of peak)",
					chunk, i, got[i], want[i], math.Abs(got[i]-want[i])/scale)
			}
		}
	}
}

// The output must not depend on the chunk size at all. This is what lets O2
// sweep chunk length to study coupling error at the Isaac-to-seismic interface
// without the synthesis itself changing underneath the experiment.
func TestOutputIsIndependentOfChunkSize(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	impulse := make([]float64, 333)
	for i := range impulse {
		impulse[i] = math.Exp(-float64(i)/80) * math.Cos(float64(i)*0.4)
	}
	in := make([]float64, 2048)
	for i := range in {
		in[i] = rng.NormFloat64()
	}

	ref, err := NewConvolver(impulse, 64)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.ProcessAll(in)

	var scale float64
	for _, v := range want {
		scale = math.Max(scale, math.Abs(v))
	}
	for _, chunk := range []int{4, 32, 100, 256} {
		c, _ := NewConvolver(impulse, chunk)
		got := c.ProcessAll(in)
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-9*scale {
				t.Fatalf("chunk=%d differs from chunk=64 at sample %d: %g vs %g", chunk, i, got[i], want[i])
			}
		}
	}
}

// The specific bug, made visible. A convolver that forgets its state between
// chunks produces a large step at every boundary; one that keeps it produces
// none. Measuring the jump at boundaries against the jump elsewhere is how a
// detector would see it, so it is how the test looks for it.
func TestNoDiscontinuityAtChunkBoundaries(t *testing.T) {
	const chunk = 32
	impulse := make([]float64, 400)
	for i := range impulse {
		impulse[i] = math.Exp(-float64(i)/90) * math.Sin(float64(i)*0.3)
	}
	// A smooth input, so any step in the output came from the chunking.
	in := make([]float64, 2048)
	for i := range in {
		in[i] = math.Sin(2 * math.Pi * 5 * float64(i) / 2000)
	}

	c, err := NewConvolver(impulse, chunk)
	if err != nil {
		t.Fatal(err)
	}
	out := c.ProcessAll(in)

	// Sample-to-sample differences at chunk boundaries versus everywhere else.
	var atBoundary, elsewhere float64
	for i := 1; i < len(out); i++ {
		d := math.Abs(out[i] - out[i-1])
		if i%chunk == 0 {
			atBoundary = math.Max(atBoundary, d)
		} else {
			elsewhere = math.Max(elsewhere, d)
		}
	}
	if atBoundary > 1.5*elsewhere {
		t.Errorf("largest step at a chunk boundary is %g, against %g elsewhere: the chunking is visible in the signal",
			atBoundary, elsewhere)
	}
}

// The same, with the real Green's function rather than a stand-in, so the
// guarantee holds for the impulse responses the model actually produces —
// which are far longer and far more sharply peaked than a synthetic decay.
func TestChunkedSynthesisMatchesMonolithicForRealGreensFunction(t *testing.T) {
	const (
		fs    = 2000.0
		chunk = 20 // 10 ms, the WP2 default
	)
	g := green.HalfSpaceGF{Soil: soil.Loam()}
	impulse, err := g.ImpulseResponse(fs, 10, 3000)
	if err != nil {
		t.Fatal(err)
	}

	// A footstep-shaped input.
	in := make([]float64, 4000)
	for i := range 1240 {
		tau := float64(i) / 1240
		in[i] = 800 * math.Sin(math.Pi*tau) * (1 + 0.3*math.Cos(4*math.Pi*tau))
	}

	want := Monolithic(in, impulse)
	c, err := NewConvolver(impulse, chunk)
	if err != nil {
		t.Fatal(err)
	}
	got := c.ProcessAll(in)

	var scale float64
	for _, v := range want {
		scale = math.Max(scale, math.Abs(v))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9*scale {
			t.Fatalf("sample %d: chunked %g, monolithic %g", i, got[i], want[i])
		}
	}
	if c.Partitions() != (len(impulse)+chunk-1)/chunk {
		t.Errorf("partitions = %d, want %d", c.Partitions(), (len(impulse)+chunk-1)/chunk)
	}
}

// The other half of V12: the offline and streaming paths must produce the same
// trace, not two approximations that have to be reconciled. Since Synthesise
// now runs through the same causal impulse response the streaming path uses,
// this is exact rather than bounded.
//
// It was not always. Synthesise originally multiplied spectra directly, and
// that disagreed with the streaming path by about 1% of peak in the coda. The
// cause turned out to be the acausal pre-ring of the band limit — a
// band-limited causal system is not causal, and a zero-phase cut at Nyquist
// rings both ways in time. Adding the discarded negative-time taps back
// reproduced the spectral answer to 3e-14, which is how the attribution was
// confirmed. Routing both paths through the causal impulse response removed it.
func TestOfflineAndStreamingPathsAgree(t *testing.T) {
	const fs = 2000.0
	g := green.HalfSpaceGF{Soil: soil.Loam()}
	const r units.Metres = 10

	force := make([]units.Newtons, 3000)
	raw := make([]float64, len(force))
	for i := range 1240 {
		tau := float64(i) / 1240
		v := 800 * math.Sin(math.Pi*tau)
		force[i], raw[i] = units.Newtons(v), v
	}

	full, err := g.Synthesise(force, fs, r)
	if err != nil {
		t.Fatal(err)
	}
	impulse, err := g.ImpulseResponse(fs, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	approx := Monolithic(raw, impulse)

	var scale float64
	for _, v := range full[:len(approx)] {
		scale = math.Max(scale, math.Abs(float64(v)))
	}
	var worst float64
	for i := range approx {
		worst = math.Max(worst, math.Abs(approx[i]-float64(full[i])))
	}
	if rel := worst / scale; rel > 1e-12 {
		t.Errorf("offline and streaming paths differ by %g of peak; they share an impulse response and should agree to roundoff", rel)
	}
}

// Reset must return the filter to a clean state, or a second run would be
// contaminated by the first — which matters for O4, where thousands of
// randomised traces get generated in one process.
func TestResetClearsState(t *testing.T) {
	impulse := []float64{1, 0.5, 0.25, 0.125}
	in := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	c, err := NewConvolver(impulse, 4)
	if err != nil {
		t.Fatal(err)
	}
	first := c.ProcessAll(in)
	c.Reset()
	second := c.ProcessAll(in)

	for i := range first {
		if math.Abs(first[i]-second[i]) > 1e-12 {
			t.Fatalf("sample %d differs after Reset: %g then %g", i, first[i], second[i])
		}
	}
}

func TestRejectsBadArguments(t *testing.T) {
	if _, err := NewConvolver([]float64{1}, 0); err == nil {
		t.Error("expected an error for a zero chunk size")
	}
	if _, err := NewConvolver(nil, 8); err == nil {
		t.Error("expected an error for an empty impulse response")
	}
	c, _ := NewConvolver([]float64{1, 2}, 4)
	if err := c.Process(make([]float64, 4), make([]float64, 3)); err == nil {
		t.Error("expected an error for a short input chunk")
	}
	if err := c.Process(make([]float64, 3), make([]float64, 4)); err == nil {
		t.Error("expected an error for a short output chunk")
	}
}

// The hot path must not allocate. A chunk arrives every few milliseconds for
// the life of the simulation, and garbage generated at that rate would show up
// as jitter in exactly the real-time factor O2 is trying to measure.
func TestProcessDoesNotAllocate(t *testing.T) {
	const chunk = 20
	impulse := make([]float64, 2000)
	for i := range impulse {
		impulse[i] = math.Exp(-float64(i) / 300)
	}
	c, err := NewConvolver(impulse, chunk)
	if err != nil {
		t.Fatal(err)
	}
	in, out := make([]float64, chunk), make([]float64, chunk)

	if got := testing.AllocsPerRun(200, func() {
		_ = c.Process(out, in)
	}); got != 0 {
		t.Errorf("Process allocates %g times per call, want 0", got)
	}
}

// The benchmark that matters for WP2: how much of a real-time budget one
// receiver's synthesis costs. Reported per chunk; a 10 ms chunk at 2 kHz has
// 10 ms of wall clock to be produced in.
func BenchmarkProcess(b *testing.B) {
	for _, chunk := range []int{10, 20, 50, 100, 200} {
		b.Run(msString(chunk), func(b *testing.B) {
			g := green.HalfSpaceGF{Soil: soil.Loam()}
			impulse, err := g.ImpulseResponse(2000, 10, 0)
			if err != nil {
				b.Fatal(err)
			}
			c, err := NewConvolver(impulse, chunk)
			if err != nil {
				b.Fatal(err)
			}
			in, out := make([]float64, chunk), make([]float64, chunk)
			b.ResetTimer()
			for b.Loop() {
				_ = c.Process(out, in)
			}
		})
	}
}

func msString(chunk int) string {
	switch chunk {
	case 10:
		return "5ms"
	case 20:
		return "10ms"
	case 50:
		return "25ms"
	case 100:
		return "50ms"
	default:
		return "100ms"
	}
}
