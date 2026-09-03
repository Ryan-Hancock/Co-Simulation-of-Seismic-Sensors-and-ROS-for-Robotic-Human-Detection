// Package dsp is the signal-processing floor the forward model stands on:
// a real FFT, linear convolution, and the resampling the sensor chain needs.
//
// Nothing here is novel, which is the point. Every physical package above it
// assumes convolution is exact and Parseval holds; if that assumption is
// wrong the error appears as a plausible-looking waveform somewhere else
// entirely. So this package is small, and it is tested against closed forms.
package dsp

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// NextPow2 returns the smallest power of two >= n. FFT sizes are kept to
// powers of two: gonum handles arbitrary lengths, but the padded sizes here
// are chosen by us, so there is no reason to make it work harder.
func NextPow2(n int) int {
	if n < 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// RFFT is the forward real-to-complex transform of seq, returning the
// len(seq)/2+1 non-redundant coefficients.
func RFFT(seq []float64) []complex128 {
	fft := fourier.NewFFT(len(seq))
	return fft.Coefficients(nil, seq)
}

// IRFFT inverts RFFT for a transform of length n.
//
// gonum's Sequence is unnormalised — it returns n times the original — so the
// 1/n belongs here rather than in every caller.
func IRFFT(coeff []complex128, n int) []float64 {
	fft := fourier.NewFFT(n)
	out := fft.Sequence(nil, coeff)
	inv := 1 / float64(n)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// Convolve returns the full linear convolution of a and b, length
// len(a)+len(b)-1, computed through the frequency domain.
//
// The zero-padding to len(a)+len(b)-1 before transforming is what makes this
// linear rather than circular. Padding to the FFT size alone is the classic
// error: the tail of the response wraps onto the head of the output, which in
// this project would put late Rayleigh energy in front of the P arrival and
// look like a causality violation in the physics rather than a bug here.
func Convolve(a, b []float64) []float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	full := len(a) + len(b) - 1
	n := NextPow2(full)

	pa := make([]float64, n)
	copy(pa, a)
	pb := make([]float64, n)
	copy(pb, b)

	fft := fourier.NewFFT(n)
	ca := fft.Coefficients(nil, pa)
	cb := fft.Coefficients(nil, pb)
	for i := range ca {
		ca[i] *= cb[i]
	}
	out := fft.Sequence(nil, ca)
	inv := 1 / float64(n)
	for i := range out {
		out[i] *= inv
	}
	return out[:full]
}

// ConvolveDirect is the O(n*m) definition of linear convolution. It exists to
// test Convolve against, and is not used in the runtime path.
func ConvolveDirect(a, b []float64) []float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	out := make([]float64, len(a)+len(b)-1)
	for i, av := range a {
		if av == 0 {
			continue
		}
		for j, bv := range b {
			out[i+j] += av * bv
		}
	}
	return out
}

// Energy is the sum of squares of x — the left-hand side of Parseval.
func Energy(x []float64) float64 {
	var s float64
	for _, v := range x {
		s += v * v
	}
	return s
}

// SpectralEnergy is the same quantity computed from a one-sided spectrum of a
// length-n real transform, with the interior bins counted twice for the
// conjugate half that RFFT does not return.
func SpectralEnergy(coeff []complex128, n int) float64 {
	var s float64
	for k, c := range coeff {
		m := real(c)*real(c) + imag(c)*imag(c)
		// DC is never mirrored; Nyquist is only present, and unmirrored,
		// when n is even.
		if k == 0 || (n%2 == 0 && k == len(coeff)-1) {
			s += m
		} else {
			s += 2 * m
		}
	}
	return s / float64(n)
}

// FreqBins returns the frequencies, in Hz, of the coefficients RFFT produces
// for a length-n transform sampled at rate fs.
func FreqBins(n int, fs float64) []float64 {
	out := make([]float64, n/2+1)
	df := fs / float64(n)
	for i := range out {
		out[i] = float64(i) * df
	}
	return out
}

// Resample changes the sample rate of x from in to out by band-limited
// sinc interpolation over a Lanczos window.
//
// The forward model computes on whichever grid the physics wants and the
// sensor reports on whichever grid the DAQ wants; those are rarely the same,
// and linear interpolation between them would alias the heel-strike transient,
// which is the part of the signal that matters most.
func Resample(x []float64, in, out float64, a int) ([]float64, error) {
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("dsp: sample rates must be positive, got in=%g out=%g", in, out)
	}
	if a < 1 {
		return nil, fmt.Errorf("dsp: lanczos half-width must be >= 1, got %d", a)
	}
	ratio := out / in
	n := int(math.Round(float64(len(x)) * ratio))
	dst := make([]float64, n)
	// Downsampling needs the kernel widened to the output Nyquist, or the
	// content between the two Nyquists folds back as alias.
	cut := math.Min(1, ratio)
	for i := range dst {
		center := float64(i) / ratio
		lo := int(math.Ceil(center - float64(a)/cut))
		hi := int(math.Floor(center + float64(a)/cut))
		var sum, wsum float64
		for j := lo; j <= hi; j++ {
			if j < 0 || j >= len(x) {
				continue
			}
			w := lanczos((center-float64(j))*cut, a)
			sum += x[j] * w
			wsum += w
		}
		if wsum != 0 {
			dst[i] = sum / wsum
		}
	}
	return dst, nil
}

func lanczos(x float64, a int) float64 {
	if x == 0 {
		return 1
	}
	af := float64(a)
	if x <= -af || x >= af {
		return 0
	}
	px := math.Pi * x
	return af * math.Sin(px) * math.Sin(px/af) / (px * px)
}
