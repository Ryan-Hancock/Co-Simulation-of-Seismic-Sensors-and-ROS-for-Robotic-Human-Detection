// Package stream turns the forward model into something that can run inside a
// co-simulation loop: force arrives in small chunks, ground velocity leaves in
// small chunks, and the result is identical to convolving the whole record at
// once.
//
// That identity is the whole problem. A geophone samples at 1-2 kHz and Isaac
// ticks at 60-240 Hz, so the seismic stream has to be produced in pieces; but
// the Green's function is seconds long, so each piece depends on force applied
// long before it. Convolving each chunk independently and concatenating gives a
// discontinuity at every boundary — and a discontinuity in ground velocity is
// exactly what a footstep looks like. WP3's detector would find them, at
// perfectly regular intervals, and there would be nothing in the waveform to
// say they were not real.
//
// So the convolution carries state across boundaries, and the test that it
// does is exact rather than approximate: chunked output must equal monolithic
// output to machine precision, for every chunk size.
package stream

import (
	"fmt"

	"gonum.org/v1/gonum/dsp/fourier"

	"geosim.dev/geosim/internal/dsp"
)

// Convolver applies a fixed impulse response to an arbitrarily long input,
// delivered a chunk at a time.
//
// The algorithm is uniformly partitioned overlap-add: the response is cut into
// blocks the size of one chunk, each transformed once at construction, and each
// arriving chunk is transformed once and accumulated against all of them
// through a frequency-domain delay line.
//
// The obvious alternative — plain overlap-add with one transform the length of
// the whole response per chunk — is simpler and fast enough at any single chunk
// size. It was rejected because its cost per second of audio scales as
// (L log L)/C: halving the chunk size doubles the work. Chunk length is not a
// constant here, it is the knob O2 sweeps to study coupling error at the
// Isaac-to-seismic interface, and a synthesis cost that blows up as that knob
// turns would make it impossible to separate coupling error from compute cost
// in the results. Partitioned convolution's cost is flat in chunk size, so the
// sweep measures what it is supposed to measure.
//
// A Convolver is not safe for concurrent use; give each receiver its own.
type Convolver struct {
	chunk   int // C, samples per chunk
	fftSize int // N = 2C, so each partial convolution fits without wrapping

	fft *fourier.FFT

	// blocks holds the transformed partitions of the impulse response.
	blocks [][]complex128
	// fdl is the frequency-domain delay line of past input spectra, one slot
	// per partition, used circularly.
	fdl    [][]complex128
	fdlPos int

	// tail is the second half of the previous chunk's convolution, waiting to
	// be added into the next. This is the state that makes the chunking exact.
	tail []float64

	// scratch buffers, reused so the hot path does not allocate.
	inBuf  []float64
	acc    []complex128
	outBuf []float64
}

// NewConvolver prepares a Convolver for the given impulse response and chunk
// size. The response is zero-padded to a whole number of partitions.
func NewConvolver(impulse []float64, chunk int) (*Convolver, error) {
	if chunk <= 0 {
		return nil, fmt.Errorf("stream: chunk size must be positive, got %d", chunk)
	}
	if len(impulse) == 0 {
		return nil, fmt.Errorf("stream: impulse response must not be empty")
	}

	n := 2 * chunk
	parts := (len(impulse) + chunk - 1) / chunk

	c := &Convolver{
		chunk:   chunk,
		fftSize: n,
		fft:     fourier.NewFFT(n),
		blocks:  make([][]complex128, parts),
		fdl:     make([][]complex128, parts),
		tail:    make([]float64, chunk),
		inBuf:   make([]float64, n),
		acc:     make([]complex128, n/2+1),
		outBuf:  make([]float64, n),
	}

	// Each partition is C samples of the response, zero-padded to 2C. The
	// padding is what keeps each partial convolution linear: C samples against
	// C samples spans 2C-1, which fits, so nothing wraps.
	block := make([]float64, n)
	for p := range parts {
		clear(block)
		copy(block, impulse[p*chunk:min((p+1)*chunk, len(impulse))])
		c.blocks[p] = c.fft.Coefficients(nil, block)
		c.fdl[p] = make([]complex128, n/2+1)
	}
	return c, nil
}

// ChunkSize is the number of samples per call to Process.
func (c *Convolver) ChunkSize() int { return c.chunk }

// Partitions is how many blocks the impulse response was cut into.
func (c *Convolver) Partitions() int { return len(c.blocks) }

// Process consumes one chunk of input and writes one chunk of output. in and
// out must both be exactly ChunkSize long; they may alias.
//
// Every call advances the filter's state, so calls must be made in order and
// none may be skipped: a gap in the input is not the same as a chunk of
// zeros, and the Convolver has no way to tell the difference.
func (c *Convolver) Process(out, in []float64) error {
	if len(in) != c.chunk || len(out) != c.chunk {
		return fmt.Errorf("stream: chunk length mismatch: in %d, out %d, want %d", len(in), len(out), c.chunk)
	}

	// Transform this chunk, zero-padded to the partition size, into the delay
	// line's current slot.
	clear(c.inBuf)
	copy(c.inBuf, in)
	// gonum fills dst in place and requires it at exactly the coefficient
	// length, so the delay-line slot is handed over whole. This is what keeps
	// the hot path free of allocation.
	c.fft.Coefficients(c.fdl[c.fdlPos], c.inBuf)

	// Accumulate the input from p chunks ago against partition p. Every term
	// begins at this chunk's start time, which is what makes a single
	// overlap-add sufficient no matter how long the response is.
	clear(c.acc)
	for p := range c.blocks {
		src := c.fdl[(c.fdlPos-p+len(c.fdl)*2)%len(c.fdl)]
		h := c.blocks[p]
		for k := range c.acc {
			c.acc[k] += src[k] * h[k]
		}
	}
	c.fdlPos = (c.fdlPos + 1) % len(c.fdl)

	// Back to time. gonum's inverse is unnormalised.
	c.fft.Sequence(c.outBuf, c.acc)
	inv := 1 / float64(c.fftSize)

	// First half plus the tail held over from last time; second half becomes
	// the new tail.
	for i := range c.chunk {
		out[i] = c.outBuf[i]*inv + c.tail[i]
	}
	for i := range c.chunk {
		c.tail[i] = c.outBuf[c.chunk+i] * inv
	}
	return nil
}

// Reset returns the Convolver to its initial state, as though no input had
// been seen. The impulse response is kept.
func (c *Convolver) Reset() {
	for _, s := range c.fdl {
		clear(s)
	}
	c.fdlPos = 0
	clear(c.tail)
}

// ProcessAll runs a whole signal through in chunks, padding the input to a
// whole number of chunks and returning the same number of samples it was
// given. It exists for testing and offline use; the runtime path calls Process.
func (c *Convolver) ProcessAll(in []float64) []float64 {
	chunks := (len(in) + c.chunk - 1) / c.chunk
	out := make([]float64, chunks*c.chunk)
	src := make([]float64, chunks*c.chunk)
	copy(src, in)
	for k := range chunks {
		lo := k * c.chunk
		_ = c.Process(out[lo:lo+c.chunk], src[lo:lo+c.chunk])
	}
	return out[:len(in)]
}

// Monolithic is the same convolution done in one piece, for tests to compare
// against. The output is truncated to the input length, matching what a
// streaming filter can have produced by the time the input ends.
func Monolithic(in, impulse []float64) []float64 {
	full := dsp.Convolve(in, impulse)
	if len(full) > len(in) {
		full = full[:len(in)]
	}
	return full
}
