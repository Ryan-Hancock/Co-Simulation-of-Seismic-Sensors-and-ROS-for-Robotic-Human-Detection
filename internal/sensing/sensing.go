// Package sensing assembles the forward model into a streaming sensor: a gait
// goes in, geophone chunks come out, indefinitely and in real time.
//
// Two things collapse into one filter here. Propagation and the sensor are
// both linear and time-invariant, so their frequency responses multiply, and
// the product inverts to a single causal impulse response from force in newtons
// to volts at the terminals. One convolution does the work of two, and — more
// usefully — there is one truncation and one causality argument rather than a
// pair that have to be reconciled.
//
// What does not collapse is the geometry. Every footfall lands somewhere
// different, so every one has its own range, its own travel time and its own
// impulse response. That is not an inconvenience to be averaged away: the
// changing range is what produces the rise and fall of a walk-past, which is
// the signature WP4 goes to the field to measure. So each contact gets its own
// voice, and the engine sums them.
package sensing

import (
	"fmt"
	"math"

	"geosim.dev/geosim/internal/config"
	"geosim.dev/geosim/internal/dsp"
	"geosim.dev/geosim/internal/green"
	"geosim.dev/geosim/internal/source"
	"geosim.dev/geosim/internal/stream"
	"geosim.dev/geosim/internal/units"
)

// RangeQuantum is how finely impulse responses are cached by range, in metres.
//
// Building one costs a few milliseconds, more than a whole chunk's budget, so
// responses are shared between contacts at similar ranges rather than computed
// per contact. The cache is keyed by range and so is bounded by the number of
// distinct footfalls, not by the span of the walk — which is why the quantum
// can afford to be fine.
//
// And it needs to be. The dominant error from coarse quantisation is not
// amplitude — that goes as 1/sqrt(r) and barely moves — but *arrival time*:
// a quantum of range is a quantum divided by the Rayleigh velocity of delay,
// and WP3 localises from arrival times. At 0.02 m over soil that is about a
// tenth of a millisecond, or two centimetres of localisation error, safely
// under the uncertainty in where a foot actually lands. At 0.1 m it would have
// been ten centimetres, which is not.
//
// This is a stand-in for the Green's function bank of slice 5, which replaces
// it with interpolation over a proper grid and quantifies the error alongside
// WP2's coupling error.
const RangeQuantum = 0.02

// Engine produces geophone output a chunk at a time.
//
// Not safe for concurrent use: it carries filter state that must advance in
// step with the samples emitted. One Engine per receiver.
type Engine struct {
	fs       float64
	chunk    int
	schedule source.Schedule
	rx, ry   float64

	gf     green.HalfSpaceGF
	sensor responder
	irLen  int
	bank   map[int]*pair

	voices  []*voice
	emitted int64
	// pending is how far ahead contacts have been scheduled, in samples.
	scheduled units.Seconds

	forceZ, forceR []float64
	outZ, outR     []float64
	accum          []float64
}

// responder is the sensor's frequency response, kept as an interface so the
// engine does not care whether it is a geophone or something else later.
type responder interface {
	Response(f units.Hertz) complex128
}

// pair is the two impulse responses one range needs: the response to a
// vertical force, and to a horizontal force along the source-receiver line.
// They are separate filters because the ratio between vertical and horizontal
// force varies through a stance, so they cannot be folded together in advance.
type pair struct{ vertical, radial []float64 }

// voice is one contact being convolved. It outlives the contact's force by the
// length of the impulse response, because the coda is still arriving.
type voice struct {
	contact  source.Contact
	ux, uy   float64 // unit vector from contact to receiver
	convZ    *stream.Convolver
	convR    *stream.Convolver
	silentAt int64 // sample index after which this voice can contribute nothing
}

// NewEngine builds the streaming sensor described by a resolved config,
// driven by a schedule, emitting chunk samples per call to Next.
//
// Impulse responses for the ranges the schedule will visit are built here
// rather than on demand, so no real-time chunk ever pays for one.
func NewEngine(res config.Resolved, sch source.Schedule, chunk int) (*Engine, error) {
	if chunk <= 0 {
		return nil, fmt.Errorf("sensing: chunk size must be positive, got %d", chunk)
	}
	if sch == nil {
		return nil, fmt.Errorf("sensing: a schedule is required")
	}
	if err := res.Sensor.Validate(); err != nil {
		return nil, err
	}
	cr, err := res.Soil.RayleighVelocity()
	if err != nil {
		return nil, err
	}

	fs := res.Sampling.Rate
	// Long enough for the furthest travel time plus two seconds of coda, which
	// puts the truncation about 80 dB down. One length for every range, so an
	// O2 sweep over chunk size is not also changing the filter.
	_, far, ok := source.RangeSpan(sch, 0, 0, 0, res.WalkDuration())
	if !ok {
		far = res.Geometry.Range
	}
	e := &Engine{
		fs:       fs,
		chunk:    chunk,
		schedule: sch,
		gf:       green.HalfSpaceGF{Soil: res.Soil},
		sensor:   res.Sensor,
		irLen:    int(math.Ceil((far/float64(cr) + 2) * fs)),
		bank:     map[int]*pair{},
		forceZ:   make([]float64, chunk),
		forceR:   make([]float64, chunk),
		outZ:     make([]float64, chunk),
		outR:     make([]float64, chunk),
		accum:    make([]float64, chunk),
	}
	if err := e.warm(0, res.WalkDuration()); err != nil {
		return nil, err
	}
	return e, nil
}

// quantum maps a range to its cache key.
func quantum(r float64) int { return int(math.Round(r / RangeQuantum)) }

// warm builds the impulse responses for every contact beginning in the window,
// so that Next never has to.
func (e *Engine) warm(from, to units.Seconds) error {
	for _, c := range e.schedule.Contacts(from, to) {
		if _, err := e.responseFor(math.Hypot(e.rx-c.X, e.ry-c.Y)); err != nil {
			return err
		}
	}
	return nil
}

// responseFor returns the cached impulse-response pair for a range, building
// it if this is the first contact to need one nearby.
func (e *Engine) responseFor(r float64) (*pair, error) {
	k := quantum(r)
	if p, ok := e.bank[k]; ok {
		return p, nil
	}
	rq := units.Metres(math.Max(float64(k)*RangeQuantum, RangeQuantum))

	vertical, err := dsp.CausalImpulseResponse(e.irLen, e.fs, func(f float64) (complex128, error) {
		h, err := e.gf.VelocityResponse(rq, units.Hertz(f))
		if err != nil {
			return 0, err
		}
		return h * e.sensor.Response(units.Hertz(f)), nil
	})
	if err != nil {
		return nil, err
	}
	radial, err := dsp.CausalImpulseResponse(e.irLen, e.fs, func(f float64) (complex128, error) {
		h, err := e.gf.RadialForceResponse(rq, units.Hertz(f))
		if err != nil {
			return 0, err
		}
		return h * e.sensor.Response(units.Hertz(f)), nil
	})
	if err != nil {
		return nil, err
	}

	p := &pair{vertical: vertical, radial: radial}
	e.bank[k] = p
	return p, nil
}

// ChunkSize is the number of samples Next produces.
func (e *Engine) ChunkSize() int { return e.chunk }

// SampleRate is the output sample rate in hertz.
func (e *Engine) SampleRate() float64 { return e.fs }

// ChunkDuration is how much simulated time one chunk covers.
func (e *Engine) ChunkDuration() units.Seconds { return units.Seconds(float64(e.chunk) / e.fs) }

// StartTime is the simulated time of the first sample of the next chunk.
//
// Derived from the sample count rather than from a clock, so sample times stay
// exactly on the sampling grid however unevenly the calling timer fires.
func (e *Engine) StartTime() units.Seconds { return units.Seconds(float64(e.emitted) / e.fs) }

// ActiveVoices is how many contacts are currently contributing, including
// those whose force has ended but whose coda is still arriving.
func (e *Engine) ActiveVoices() int { return len(e.voices) }

// Next fills out with the next chunk of geophone output in volts. out must be
// exactly ChunkSize long.
//
// Calls must be in order and none may be skipped: every voice's convolver has
// no way to distinguish a gap in the stream from a chunk of silence.
func (e *Engine) Next(out []float32) error {
	if len(out) != e.chunk {
		return fmt.Errorf("sensing: output length %d, want %d", len(out), e.chunk)
	}
	t0 := units.Seconds(float64(e.emitted) / e.fs)
	t1 := t0 + e.ChunkDuration()

	if err := e.admit(t0, t1); err != nil {
		return err
	}

	clear(e.accum)
	for _, v := range e.voices {
		if e.emitted > v.silentAt {
			continue
		}
		for i := range e.chunk {
			f := v.contact.ForceAt(t0 + units.Seconds(float64(i)/e.fs))
			e.forceZ[i] = f[2]
			// Only the component along the source-receiver line drives the
			// vertical motion a geophone sees; the transverse part goes into
			// Love waves, which have no vertical component at all.
			e.forceR[i] = f[0]*v.ux + f[1]*v.uy
		}
		if err := v.convZ.Process(e.outZ, e.forceZ); err != nil {
			return err
		}
		if err := v.convR.Process(e.outR, e.forceR); err != nil {
			return err
		}
		for i := range e.chunk {
			e.accum[i] += e.outZ[i] + e.outR[i]
		}
	}
	e.retire()

	for i := range out {
		out[i] = float32(e.accum[i])
	}
	e.emitted += int64(e.chunk)
	return nil
}

// admit creates voices for contacts beginning in this chunk, and keeps the
// response bank warmed a little way ahead.
//
// For a configured walk NewEngine has already warmed the whole run, so this
// never builds anything. The lookahead is the fallback for a schedule that
// outlives the configured walk, and it *can* stall a chunk: building several
// responses takes tens of milliseconds. Slice 5's bank, sized once up front,
// is what removes the possibility rather than merely making it unlikely.
func (e *Engine) admit(from, to units.Seconds) error {
	const lookahead = 2 * units.Seconds(1)
	if to+lookahead > e.scheduled {
		if err := e.warm(e.scheduled, to+lookahead); err != nil {
			return err
		}
		e.scheduled = to + lookahead
	}
	for _, c := range e.schedule.Contacts(from, to) {
		dx, dy := e.rx-c.X, e.ry-c.Y
		r := math.Hypot(dx, dy)
		if r <= 0 {
			return fmt.Errorf("sensing: contact %q at the receiver's own position", c.ID)
		}
		p, err := e.responseFor(r)
		if err != nil {
			return err
		}
		convZ, err := stream.NewConvolver(p.vertical, e.chunk)
		if err != nil {
			return err
		}
		convR, err := stream.NewConvolver(p.radial, e.chunk)
		if err != nil {
			return err
		}
		// The voice must keep running until its own force has ended and the
		// impulse response it excited has run out. Retiring it at toe-off
		// would truncate the coda mid-arrival, which at range is most of the
		// signal.
		silent := int64(math.Ceil(float64(c.End())*e.fs)) + int64(e.irLen)
		e.voices = append(e.voices, &voice{
			contact: c, ux: dx / r, uy: dy / r,
			convZ: convZ, convR: convR, silentAt: silent,
		})
	}
	return nil
}

// retire drops voices that can no longer contribute.
func (e *Engine) retire() {
	kept := e.voices[:0]
	for _, v := range e.voices {
		if e.emitted <= v.silentAt {
			kept = append(kept, v)
		}
	}
	e.voices = kept
}

// Reference synthesises n samples of the same signal an Engine produces, in
// one piece rather than a chunk at a time.
//
// It is the oracle for the streaming path. Producing the two by visibly
// different routes — one monolithic convolution per contact against a
// partitioned streaming filter driven by a timer — is what makes their
// agreement evidence rather than a tautology.
func Reference(res config.Resolved, sch source.Schedule, n int) ([]float64, error) {
	e, err := NewEngine(res, sch, 1)
	if err != nil {
		return nil, err
	}
	fs := res.Sampling.Rate
	span := units.Seconds(float64(n) / fs)

	out := make([]float64, n)
	for _, c := range sch.Contacts(0, span) {
		dx, dy := -c.X, -c.Y
		r := math.Hypot(dx, dy)
		p, err := e.responseFor(r)
		if err != nil {
			return nil, err
		}
		ux, uy := dx/r, dy/r

		fz := make([]float64, n)
		fr := make([]float64, n)
		for i := range n {
			f := c.ForceAt(units.Seconds(float64(i) / fs))
			fz[i] = f[2]
			fr[i] = f[0]*ux + f[1]*uy
		}
		vz := stream.Monolithic(fz, p.vertical)
		vr := stream.Monolithic(fr, p.radial)
		for i := range out {
			out[i] += vz[i] + vr[i]
		}
	}
	return out, nil
}
