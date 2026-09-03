// Package sensing assembles the forward model into a streaming sensor: a gait
// goes in, geophone chunks come out, indefinitely and in real time.
//
// The whole chain collapses into one filter. Propagation and the sensor are
// both linear and time-invariant, so their frequency responses multiply, and
// the product inverts to a single causal impulse response from force in
// newtons to volts at the terminals. One convolution then does the work of
// two, and — more usefully — there is one truncation and one causality
// argument rather than a pair that have to be reconciled.
//
// The source is the part that is not LTI, and it is kept outside: forces are
// generated per sample from gait phase and fed in.
package sensing

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/geophone"
	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/grf"
	"geosim.dev/geosim/internal/stream"
	"geosim.dev/geosim/internal/units"
)

// Engine produces geophone output a chunk at a time.
//
// Not safe for concurrent use: it carries filter state that must advance in
// step with the samples it has emitted. One Engine per receiver.
type Engine struct {
	fs       float64
	chunk    int
	conv     *stream.Convolver
	gait     *Gait
	emitted  int64
	forceBuf []float64
	voltBuf  []float64
}

// Gait emits the vertical force of a walker whose steps repeat at a fixed
// cadence.
//
// One foot, in slice 0's sense: a single stance profile repeated. Two feet at
// their own positions, the heel-strike transient, and forces arriving from
// Isaac over ROS all belong to slice 2. What this is for is giving the
// transport something continuous and recognisable to carry.
type Gait struct {
	Stance grf.Stance
	// Period is the interval between successive heel strikes of this foot —
	// one gait cycle, not one step. Zero derives it from the stance duration
	// on the assumption that stance is 60% of the cycle.
	Period units.Seconds
}

func (g Gait) period() float64 {
	if g.Period > 0 {
		return float64(g.Period)
	}
	return float64(g.Stance.Duration) / grf.StanceFraction
}

// ForceAt is the vertical force at absolute time t.
func (g Gait) ForceAt(t float64) units.Newtons {
	p := g.period()
	phase := math.Mod(t, p)
	if phase < 0 {
		phase += p
	}
	return g.Stance.ForceAt(units.Seconds(phase))
}

// NewEngine builds the streaming sensor described by a resolved config,
// emitting chunk samples per call to Next.
func NewEngine(res config.Resolved, chunk int) (*Engine, error) {
	if chunk <= 0 {
		return nil, fmt.Errorf("sensing: chunk size must be positive, got %d", chunk)
	}
	fs := res.Sampling.Rate
	gf := green.HalfSpaceGF{Soil: res.Soil}
	r := units.Metres(res.Geometry.Range)

	if err := res.Sensor.Validate(); err != nil {
		return nil, err
	}
	cr, err := res.Soil.RayleighVelocity()
	if err != nil {
		return nil, err
	}

	// Long enough for the travel time plus two seconds of coda, which puts the
	// truncation about 80 dB down. Fixed independently of chunk size: an O2
	// sweep over chunk length must not be changing the filter underneath the
	// experiment.
	length := int(math.Ceil((float64(r)/float64(cr) + 2) * fs))

	impulse, err := dsp.CausalImpulseResponse(length, fs, func(f float64) (complex128, error) {
		h, err := gf.VelocityResponse(r, units.Hertz(f))
		if err != nil {
			return 0, err
		}
		return h * res.Sensor.Response(units.Hertz(f)), nil
	})
	if err != nil {
		return nil, err
	}

	conv, err := stream.NewConvolver(impulse, chunk)
	if err != nil {
		return nil, err
	}
	return &Engine{
		fs:       fs,
		chunk:    chunk,
		conv:     conv,
		gait:     &Gait{Stance: res.Walker},
		forceBuf: make([]float64, chunk),
		voltBuf:  make([]float64, chunk),
	}, nil
}

// ChunkSize is the number of samples Next produces.
func (e *Engine) ChunkSize() int { return e.chunk }

// SampleRate is the output sample rate in hertz.
func (e *Engine) SampleRate() float64 { return e.fs }

// ChunkDuration is how much simulated time one chunk covers.
func (e *Engine) ChunkDuration() units.Seconds {
	return units.Seconds(float64(e.chunk) / e.fs)
}

// StartTime is the simulated time of the first sample of the next chunk,
// relative to the Engine's own zero. Derived from the sample count rather than
// from a clock, so sample times stay exactly on the sampling grid however
// unevenly the calling timer happens to fire.
func (e *Engine) StartTime() units.Seconds {
	return units.Seconds(float64(e.emitted) / e.fs)
}

// Next fills out with the next chunk of geophone output in volts. out must be
// exactly ChunkSize long.
//
// Calls must be in order and none may be skipped: the convolver has no way to
// distinguish a gap in the stream from a chunk of silence.
func (e *Engine) Next(out []float32) error {
	if len(out) != e.chunk {
		return fmt.Errorf("sensing: output length %d, want %d", len(out), e.chunk)
	}
	t0 := float64(e.emitted) / e.fs
	for i := range e.chunk {
		e.forceBuf[i] = float64(e.gait.ForceAt(t0 + float64(i)/e.fs))
	}
	if err := e.conv.Process(e.voltBuf, e.forceBuf); err != nil {
		return err
	}
	for i, v := range e.voltBuf {
		out[i] = float32(v)
	}
	e.emitted += int64(e.chunk)
	return nil
}

// NoiseDensity is the sensor's own voltage noise floor, in V/sqrt(Hz),
// exposed so a node can report it without reaching into the sensor model.
func (e *Engine) NoiseDensity(g geophone.Geophone) float64 { return g.NoiseDensity() }

// Reference synthesises n samples of the same signal an Engine produces, in
// one piece rather than a chunk at a time.
//
// It is the oracle for the streaming path. Producing the two by visibly
// different routes — one monolithic convolution against a partitioned
// streaming filter, driven by a timer, through a transport — is what makes
// their agreement evidence rather than a tautology.
func Reference(res config.Resolved, n int) ([]float64, error) {
	fs := res.Sampling.Rate
	gf := green.HalfSpaceGF{Soil: res.Soil}
	r := units.Metres(res.Geometry.Range)
	if err := res.Sensor.Validate(); err != nil {
		return nil, err
	}
	cr, err := res.Soil.RayleighVelocity()
	if err != nil {
		return nil, err
	}
	length := int(math.Ceil((float64(r)/float64(cr) + 2) * fs))

	impulse, err := dsp.CausalImpulseResponse(length, fs, func(f float64) (complex128, error) {
		h, err := gf.VelocityResponse(r, units.Hertz(f))
		if err != nil {
			return 0, err
		}
		return h * res.Sensor.Response(units.Hertz(f)), nil
	})
	if err != nil {
		return nil, err
	}

	gait := Gait{Stance: res.Walker}
	force := make([]float64, n)
	for i := range force {
		force[i] = float64(gait.ForceAt(float64(i) / fs))
	}
	return stream.Monolithic(force, impulse), nil
}
