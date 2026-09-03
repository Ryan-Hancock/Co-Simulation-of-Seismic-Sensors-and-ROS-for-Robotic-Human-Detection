package sensing

import (
	"math"
	"testing"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/source"
	"geosim.dev/geosim/internal/units"
)

func base(t *testing.T) config.Resolved {
	t.Helper()
	c := config.Default()
	c.Sampling.Rate = 1000
	c.Geometry.ApproachLength = 6
	res, err := c.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func run(t *testing.T, res config.Resolved, sch source.Schedule, chunk, n int) []float64 {
	t.Helper()
	e, err := NewEngine(res, sch, chunk)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]float32, chunk)
	var out []float64
	for len(out) < n {
		if err := e.Next(buf); err != nil {
			t.Fatal(err)
		}
		for _, v := range buf {
			out = append(out, float64(v))
		}
	}
	return out[:n]
}

// The chunked engine must agree with the monolithic reference, exactly as the
// single-source version did — but now across several simultaneous voices, each
// with its own range, its own arrival time and its own filter state. Summing
// voices is where an off-by-one in retirement or admission would show.
func TestStreamingMatchesReferenceAcrossVoices(t *testing.T) {
	res := base(t)
	walk := res.WalkPast()
	const n = 6000

	want, err := Reference(res, walk, n)
	if err != nil {
		t.Fatal(err)
	}
	var scale float64
	for _, v := range want {
		scale = math.Max(scale, math.Abs(v))
	}
	if scale == 0 {
		t.Fatal("reference is silent")
	}
	for _, chunk := range []int{10, 25, 100} {
		got := run(t, res, walk, chunk, n)
		for i := range want {
			// Looser than the single-voice case in package stream, because
			// several voices are summed here and float64 roundoff accumulates
			// across them. Still eight orders below anything physical.
			if math.Abs(got[i]-want[i]) > 1e-7*scale {
				t.Fatalf("chunk=%d sample %d: streamed %g, reference %g", chunk, i, got[i], want[i])
			}
		}
	}
}

// Superposition. The medium is linear, so two contacts together must give
// exactly the sum of each alone. If they do not, voices are sharing state.
func TestContactsSuperpose(t *testing.T) {
	res := base(t)
	const n = 4000

	mk := func(id string, x, y float64, start units.Seconds) fixed {
		return fixed{[]source.Contact{{
			ID: id, X: x, Y: y, Start: start,
			Profile: source.Footfall{Stance: res.Walker, Heading: 0},
		}}}
	}
	a := mk("a", -3, 8, 0.1)
	b := mk("b", 2, 9, 0.4)
	both := fixed{append(append([]source.Contact{}, a.contacts...), b.contacts...)}

	ra, err := Reference(res, a, n)
	if err != nil {
		t.Fatal(err)
	}
	rb, err := Reference(res, b, n)
	if err != nil {
		t.Fatal(err)
	}
	rab, err := Reference(res, both, n)
	if err != nil {
		t.Fatal(err)
	}
	var scale float64
	for _, v := range rab {
		scale = math.Max(scale, math.Abs(v))
	}
	for i := range rab {
		if math.Abs(rab[i]-(ra[i]+rb[i])) > 1e-7*scale {
			t.Fatalf("sample %d: together %g, separately %g + %g", i, rab[i], ra[i], rb[i])
		}
	}
}

// A walk past a sensor must rise and fall, peaking near closest approach. This
// is the signature WP4 goes to the field to measure, and it comes entirely from
// the changing range — the force profile is identical at every step.
func TestWalkPastEnvelopePeaksAtClosestApproach(t *testing.T) {
	res := base(t)
	res.Geometry.Range = 6
	walk := res.WalkPast()
	fs := res.Sampling.Rate
	n := int(float64(res.WalkDuration()) * fs)

	trace := run(t, res, walk, 50, n)

	// Peak output within each footfall's window, against that footfall's range.
	type sample struct{ rng, amp float64 }
	var pts []sample
	for _, c := range walk.Contacts(0, units.Seconds(float64(n)/fs)) {
		lo := int(float64(c.Start) * fs)
		hi := min(lo+int(0.5*fs), n)
		if lo >= hi {
			continue
		}
		var amp float64
		for _, v := range trace[lo:hi] {
			amp = math.Max(amp, math.Abs(v))
		}
		pts = append(pts, sample{math.Hypot(c.X, c.Y), amp})
	}
	if len(pts) < 8 {
		t.Fatalf("only %d footfalls to judge", len(pts))
	}

	loudest, closest := 0, 0
	for i := range pts {
		if pts[i].amp > pts[loudest].amp {
			loudest = i
		}
		if pts[i].rng < pts[closest].rng {
			closest = i
		}
	}
	if abs(loudest-closest) > 2 {
		t.Errorf("loudest footfall is number %d but the closest is %d", loudest, closest)
	}
	// And it really is a rise and a fall, not a monotone drift.
	if pts[0].amp >= pts[loudest].amp || pts[len(pts)-1].amp >= pts[loudest].amp {
		t.Errorf("no walk-past envelope: first %.3g, loudest %.3g, last %.3g",
			pts[0].amp, pts[loudest].amp, pts[len(pts)-1].amp)
	}
	// Amplitude must fall with range across the whole pass.
	if pts[closest].amp <= pts[0].amp*1.1 {
		t.Errorf("closest approach is only %.3g against %.3g at the start; range is barely affecting amplitude",
			pts[closest].amp, pts[0].amp)
	}
}

// Voices must be retired once their coda has run out, or a long walk would
// accumulate filter state without bound — and each voice costs two
// convolutions in every chunk.
func TestVoicesAreRetired(t *testing.T) {
	res := base(t)
	walk := res.WalkPast()
	e, err := NewEngine(res, walk, 50)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]float32, 50)
	var most int
	for range int(float64(res.WalkDuration()) * res.Sampling.Rate / 50) {
		if err := e.Next(buf); err != nil {
			t.Fatal(err)
		}
		most = max(most, e.ActiveVoices())
	}
	// A stance is 0.62 s and the coda a couple of seconds, at about two steps
	// a second: a handful of voices, not dozens.
	if most == 0 {
		t.Fatal("no voices were ever active")
	}
	if most > 12 {
		t.Errorf("as many as %d voices active at once; retirement is not keeping up", most)
	}
}

// Sample times come from the engine's own counter, not from a clock, so they
// stay on the sampling grid no matter how the caller schedules chunks.
func TestStartTimeTracksSamplesNotCalls(t *testing.T) {
	res := base(t)
	const chunk = 32
	e, err := NewEngine(res, res.WalkPast(), chunk)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]float32, chunk)
	for i := range 20 {
		want := units.Seconds(float64(i*chunk) / res.Sampling.Rate)
		if got := e.StartTime(); math.Abs(float64(got-want)) > 1e-12 {
			t.Fatalf("chunk %d starts at %g s, want %g", i, got, want)
		}
		if err := e.Next(buf); err != nil {
			t.Fatal(err)
		}
	}
	if got, want := e.ChunkDuration(), units.Seconds(chunk/res.Sampling.Rate); math.Abs(float64(got-want)) > 1e-12 {
		t.Errorf("chunk duration %g s, want %g", got, want)
	}
}

// Impulse responses are shared between contacts at similar ranges because
// building one costs more than a chunk's whole time budget. The error that
// introduces has to be small enough not to matter, and stated rather than
// assumed.
//
// Measured as amplitude and arrival time separately, not sample by sample. A
// sharp transient shifted by a fraction of a sample differs enormously
// point-by-point while being, for every purpose this project has, the same
// waveform — and it is the arrival time that WP3 localises from, so that is
// the quantity worth bounding.
func TestRangeQuantisationErrorIsBounded(t *testing.T) {
	res := base(t)
	fs := res.Sampling.Rate
	const n = 3000
	// Energy rather than peak amplitude. The peak of a sharp transient depends
	// on where the sampling grid happens to fall relative to it, so a shift of
	// a tenth of a sample moves it by more than the physics does — that is an
	// artefact of measuring, not of the model.
	at := func(y float64) (rms float64, peakAt int) {
		sch := fixed{[]source.Contact{{
			ID: "f", X: 0, Y: y, Start: 0.05,
			Profile: source.Footfall{Stance: res.Walker, Heading: 0},
		}}}
		out, err := Reference(res, sch, n)
		if err != nil {
			t.Fatal(err)
		}
		var sum, peak float64
		for i, v := range out {
			sum += v * v
			if math.Abs(v) > peak {
				peak, peakAt = math.Abs(v), i
			}
		}
		return math.Sqrt(sum / float64(len(out))), peakAt
	}

	// Two ranges a full quantum apart are the worst case: they are the closest
	// pair guaranteed to land in different cache slots.
	p0, t0 := at(8.0)
	p1, t1 := at(8.0 + RangeQuantum)

	if rel := math.Abs(p1-p0) / p0; rel > 0.005 {
		t.Errorf("one quantum of range changes the signal's energy by %.3f%%, want under 0.5%%", rel*100)
	}
	// The true delay difference across one quantum, in samples.
	cr, err := res.Soil.RayleighVelocity()
	if err != nil {
		t.Fatal(err)
	}
	trueShift := RangeQuantum / float64(cr) * fs
	if trueShift > 1 {
		t.Errorf("one quantum is %.2f samples of delay at %g Hz; too coarse to place an arrival", trueShift, fs)
	}
	if shift := abs(t1 - t0); float64(shift) > 1+trueShift {
		t.Errorf("peak moved %d samples across one quantum, want at most %.2f", shift, 1+trueShift)
	}

	// And in metres of localisation error, which is the number that matters.
	if loc := trueShift / fs * float64(cr); loc > 0.05 {
		t.Errorf("quantisation implies %.3f m of localisation error, want under 0.05 m", loc)
	}
}

func TestRejectsBadArguments(t *testing.T) {
	res := base(t)
	if _, err := NewEngine(res, res.WalkPast(), 0); err == nil {
		t.Error("expected an error for a zero chunk size")
	}
	if _, err := NewEngine(res, nil, 16); err == nil {
		t.Error("expected an error for a nil schedule")
	}
	bad := res
	bad.Sensor.CoilResistance = 0
	if _, err := NewEngine(bad, res.WalkPast(), 16); err == nil {
		t.Error("expected an error for an invalid sensor")
	}
	e, err := NewEngine(res, res.WalkPast(), 16)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Next(make([]float32, 15)); err == nil {
		t.Error("expected an error for a wrong-sized output chunk")
	}
	// A contact directly under the receiver has no defined range.
	onTop := fixed{[]source.Contact{{
		ID: "f", X: 0, Y: 0, Start: 0.01,
		Profile: source.Footfall{Stance: grf.Walker(75)},
	}}}
	e2, err := NewEngine(res, onTop, 16)
	if err == nil {
		err = e2.Next(make([]float32, 16))
	}
	if err == nil {
		t.Error("expected an error for a contact at the receiver's position")
	}
}

// fixed is a Schedule with a hand-written contact list.
type fixed struct{ contacts []source.Contact }

func (f fixed) Contacts(from, to units.Seconds) []source.Contact {
	var out []source.Contact
	for _, c := range f.contacts {
		if c.Start >= from && c.Start < to {
			out = append(out, c)
		}
	}
	return out
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// One receiver's whole cost per chunk, with several voices live. This is the
// number that scales to a robot carrying an array.
func BenchmarkNextDuringWalk(b *testing.B) {
	c := config.Default()
	res, err := c.Resolve()
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 20
	e, err := NewEngine(res, res.WalkPast(), chunk)
	if err != nil {
		b.Fatal(err)
	}
	buf := make([]float32, chunk)
	// Advance into the middle of the walk, where the most voices are live.
	for range 800 {
		if err := e.Next(buf); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for b.Loop() {
		if err := e.Next(buf); err != nil {
			b.Fatal(err)
		}
	}
}
