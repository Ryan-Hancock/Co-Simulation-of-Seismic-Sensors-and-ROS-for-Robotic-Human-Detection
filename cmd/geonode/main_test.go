package main

import (
	"math"
	"sync"
	"testing"
	"time"

	"conductor.dev/conductor/conductortest"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/sensing"
	"geosim.dev/geosim/msgs"
)

// simClock is a clock the test moves by hand, standing in for /clock.
//
// The node's timers fire only when the harness ticks them, so nothing here
// races: the test decides both when a chunk is produced and what time it is
// produced at, which is exactly the control a co-simulation needs and exactly
// what makes this reproducible.
type simClock struct {
	mu      sync.Mutex
	now     time.Time
	started bool
}

func (c *simClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *simClock) Started() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.started
}

func (c *simClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now, c.started = c.now.Add(d), true
}

func (c *simClock) After(time.Duration) <-chan time.Time { return make(chan time.Time) }
func (c *simClock) Ticker(time.Duration) (<-chan time.Time, func()) {
	return make(chan time.Time), func() {}
}

func newNode(t *testing.T) (*Geophone, config.Resolved, int) {
	t.Helper()
	res, err := config.Default().Resolve()
	if err != nil {
		t.Fatal(err)
	}
	chunk := int(res.Sampling.Rate / chunkRateHz)
	engine, err := sensing.NewEngine(res, res.WalkPast(), chunk)
	if err != nil {
		t.Fatal(err)
	}
	return &Geophone{
		engine:  engine,
		rangeM:  res.Geometry.Range,
		samples: make([]float32, chunk),
	}, res, chunk
}

// The slice-1 deliverable, end to end: a stream published chunk by chunk over
// ROS, under simulated time, reassembles into exactly the trace the offline
// model produces in one piece.
//
// V12 is already proved inside the convolver, but this is the statement that
// matters for WP2, because it holds across everything between: the sensing
// engine's own buffering, a timer firing on a simulated clock, message
// encoding, and the transport. Any of those could have dropped, duplicated or
// reordered a chunk, and the waveform would still have looked plausible.
func TestPublishedStreamReassemblesToTheOfflineTrace(t *testing.T) {
	node, res, chunk := newNode(t)
	clock := &simClock{now: time.Unix(1000, 0), started: true}

	app := conductortest.RunWith(t, conductortest.Options{Clock: clock, SimTime: true}, node)
	rec := conductortest.Watch[msgs.GeophoneChunk](app, "geophone/chunk")

	const chunks = 250 // 2.5 s at 100 Hz, several gait cycles
	for range chunks {
		app.Tick("geophone")
		clock.advance(time.Second / chunkRateHz)
	}

	got := rec.All()
	if len(got) != chunks {
		t.Fatalf("received %d chunks, expected %d: the stream has gaps or duplicates", len(got), chunks)
	}

	var stitched []float64
	for i, c := range got {
		if c.SampleRateHz != res.Sampling.Rate {
			t.Fatalf("chunk %d reports %g Hz, want %g", i, c.SampleRateHz, res.Sampling.Rate)
		}
		if len(c.Samples) != chunk {
			t.Fatalf("chunk %d has %d samples, want %d", i, len(c.Samples), chunk)
		}
		for _, s := range c.Samples {
			stitched = append(stitched, float64(s))
		}
	}

	want, err := sensing.Reference(res, res.WalkPast(), len(stitched))
	if err != nil {
		t.Fatal(err)
	}
	var scale float64
	for _, v := range want {
		scale = math.Max(scale, math.Abs(v))
	}
	if scale == 0 {
		t.Fatal("reference trace is silent")
	}
	// float32 on the wire is the only lossy step, and deliberately so: the
	// sensor's own noise floor is far above float32 resolution at these
	// amplitudes, so the tolerance is set by the wire format rather than by
	// anything in the physics.
	for i := range want {
		if math.Abs(stitched[i]-want[i]) > 1e-6*scale {
			t.Fatalf("sample %d: streamed %g, offline %g (diff %g of peak)",
				i, stitched[i], want[i], math.Abs(stitched[i]-want[i])/scale)
		}
	}
}

// The published trace must not carry a step at every chunk boundary. This is
// the same failure V12 guards against, checked on what actually left the node:
// a regular train of steps at the chunk rate would be indistinguishable from
// footsteps to WP3's detector.
func TestNoChunkRateArtefactInThePublishedStream(t *testing.T) {
	node, _, chunk := newNode(t)
	clock := &simClock{now: time.Unix(1000, 0), started: true}
	app := conductortest.RunWith(t, conductortest.Options{Clock: clock, SimTime: true}, node)
	rec := conductortest.Watch[msgs.GeophoneChunk](app, "geophone/chunk")

	for range 200 {
		app.Tick("geophone")
		clock.advance(time.Second / chunkRateHz)
	}

	var x []float64
	for _, c := range rec.All() {
		for _, s := range c.Samples {
			x = append(x, float64(s))
		}
	}
	var atBoundary, elsewhere float64
	for i := 1; i < len(x); i++ {
		d := math.Abs(x[i] - x[i-1])
		if i%chunk == 0 {
			atBoundary = math.Max(atBoundary, d)
		} else {
			elsewhere = math.Max(elsewhere, d)
		}
	}
	if atBoundary > 1.5*elsewhere {
		t.Errorf("largest step at a chunk boundary is %g against %g elsewhere: the chunking is audible in the published signal",
			atBoundary, elsewhere)
	}
}

// Stamps must come from the simulated clock and advance by exactly one chunk.
// A stream stamped in wall time on a simulated robot is the failure that makes
// every downstream consumer's time arithmetic wrong by however far the two
// clocks happen to be apart.
func TestStampsFollowSimulatedTime(t *testing.T) {
	node, _, _ := newNode(t)
	epoch := time.Unix(1000, 0)
	clock := &simClock{now: epoch, started: true}
	app := conductortest.RunWith(t, conductortest.Options{Clock: clock, SimTime: true}, node)
	rec := conductortest.Watch[msgs.GeophoneChunk](app, "geophone/chunk")

	const step = time.Second / chunkRateHz
	for range 20 {
		app.Tick("geophone")
		clock.advance(step)
	}

	got := rec.All()
	if len(got) == 0 {
		t.Fatal("no chunks published")
	}
	for i, c := range got {
		want := epoch.Add(time.Duration(i) * step)
		if !c.Header.Stamp.Equal(want) {
			t.Errorf("chunk %d stamped %v, want %v (simulated time, not the wall)", i, c.Header.Stamp, want)
		}
		if c.Header.FrameId != "geophone_link" {
			t.Errorf("chunk %d frame id %q, want geophone_link", i, c.Header.FrameId)
		}
	}
	// And the stamps are far from the wall clock, so a test that accidentally
	// used real time would not pass by coincidence.
	if d := time.Since(got[0].Header.Stamp); d < 24*time.Hour {
		t.Errorf("simulated epoch is only %v from now; the clock may not be simulated at all", d)
	}
}

// The timer's rate tag and the chunk size are derived from different places —
// a compile-time string and a constant — so the node checks they agree and
// aborts if they do not. Nothing else would notice.
func TestTimerRateMatchesChunkRate(t *testing.T) {
	node, res, chunk := newNode(t)
	app := conductortest.RunWith(t, conductortest.Options{}, node)
	app.Tick("geophone")

	if node.Tick.Period() != time.Second/chunkRateHz {
		t.Errorf("timer period %v, want %v", node.Tick.Period(), time.Second/chunkRateHz)
	}
	if want := int(res.Sampling.Rate * node.Tick.Period().Seconds()); chunk != want {
		t.Errorf("chunk is %d samples but the timer period covers %d", chunk, want)
	}
}
